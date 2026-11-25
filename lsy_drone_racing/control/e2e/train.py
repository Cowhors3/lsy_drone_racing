"""PPO training for end-to-end drone racing, written in pure JAX.

Data collection and network updates are compiled into one function that runs entirely on the GPU.
The file is organised as:

1. Config           all hyperparameters in one place
2. Networks         small MLPs for the policy (actor) and the value estimate (critic)
3. Rollout          let the current policy fly in all environments for a few steps
4. PPO update       improve the networks from the collected data
5. Training loop    repeat rollout + update, print progress, save the policy

Run from the repository root:

    python -m lsy_drone_racing.control.e2e.train --config level2.toml

A faster policy, saved next to the default one:

    python -m lsy_drone_racing.control.e2e.train --max_speed 3.0 --time_penalty 0.02 \
        --drone_radius 0.08 --out policy_fast.npz

Continue training saved weights with different settings (``--max_tilt`` must stay the same):

    python -m lsy_drone_racing.control.e2e.train --init policy_fast.npz --time_penalty 0.03 \
        --max_speed 3.0 --drone_radius 0.08 --out policy_fast2.npz
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from lsy_drone_racing.control.e2e import env

POLICY_PATH = Path(__file__).parent / "policy.npz"

# region 1. Config


class Config(NamedTuple):
    """PPO hyperparameters."""

    n_envs: int = 2048  # Environments simulated in parallel
    n_steps: int = 64  # Steps per environment between two updates
    total_steps: int = 200_000_000
    hidden: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4  # Decays linearly to zero
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    clip: float = 0.2  # PPO clipping range
    epochs: int = 4  # Passes over each batch of collected data
    minibatches: int = 8
    value_coef: float = 0.5
    entropy_coef: float = 0.01  # Keeps exploring, without it the policy gets stuck early
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5
    seed: int = 0


# region 2. Networks


def init_mlp(key: Array, sizes: list[int], last_scale: float) -> list[tuple[Array, Array]]:
    """Create the weights of an MLP as a list of (weight, bias) pairs."""
    layers = []
    for i, k in enumerate(jax.random.split(key, len(sizes) - 1)):
        scale = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        w = jax.nn.initializers.orthogonal(scale)(k, (sizes[i], sizes[i + 1]))
        layers.append((w, jnp.zeros(sizes[i + 1])))
    return layers


def mlp(layers: list[tuple[Array, Array]], x: Array) -> Array:
    """Evaluate an MLP with tanh activations on the hidden layers."""
    for w, b in layers[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def policy_mean(actor: list[tuple[Array, Array]], obs: Array) -> Array:
    """The action the policy prefers, in [-1, 1]. This is what the deployed controller uses."""
    return jnp.tanh(mlp(actor, obs))


def init_params(key: Array, cfg: Config) -> dict:
    """Create actor, critic and the exploration noise level."""
    k_actor, k_critic = jax.random.split(key)
    return {
        "actor": init_mlp(k_actor, [env.OBS_DIM, *cfg.hidden, env.ACTION_DIM], 0.01),
        "critic": init_mlp(k_critic, [env.OBS_DIM, *cfg.hidden, 1], 1.0),
        "log_std": jnp.full(env.ACTION_DIM, cfg.init_log_std),
    }


def log_prob(params: dict, obs: Array, action: Array) -> Array:
    """Log probability of an action under the Gaussian policy."""
    std = jnp.exp(params["log_std"])
    z = (action - policy_mean(params["actor"], obs)) / std
    return jnp.sum(-0.5 * z**2 - params["log_std"] - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def save_policy(params: dict, max_tilt: float, path: Path = POLICY_PATH):
    """Save the weights together with the tilt limit they were trained for.

    The controller only needs the actor (``w*``, ``b*``) and ``max_tilt``. The critic (``cw*``,
    ``cb*``) and the noise level are stored so that training can be continued with ``--init``.
    """
    arrays = {"max_tilt": np.float32(max_tilt), "log_std": np.asarray(params["log_std"])}
    for name, prefix in (("actor", ""), ("critic", "c")):
        for i, (w, b) in enumerate(params[name]):
            arrays[f"{prefix}w{i}"], arrays[f"{prefix}b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(path, **arrays)


def read_layers(data: dict, prefix: str = "") -> list[tuple[Array, Array]]:
    """Collect the (weight, bias) pairs of one network from a saved file."""
    n = sum(key.startswith(f"{prefix}w") for key in data)
    return [
        (jnp.asarray(data[f"{prefix}w{i}"]), jnp.asarray(data[f"{prefix}b{i}"])) for i in range(n)
    ]


def load_policy(path: Path = POLICY_PATH) -> tuple[list[tuple[Array, Array]], float]:
    """Load the actor and its tilt limit (the default for files saved before it was stored)."""
    data = dict(np.load(path))
    return read_layers(data), float(data.get("max_tilt", env.MAX_TILT))


def load_params(path: Path, params: dict, max_tilt: float) -> dict:
    """Replace freshly initialized parameters by saved ones to continue a training run."""
    data = dict(np.load(path))
    saved_tilt = float(data.get("max_tilt", env.MAX_TILT))
    assert abs(saved_tilt - max_tilt) < 1e-6, f"{path.name} needs --max_tilt {saved_tilt:.2f}"
    params = params | {"actor": read_layers(data)}
    if "log_std" in data:  # Older files hold only the actor, the critic then starts from scratch
        params |= {"critic": read_layers(data, "c"), "log_std": jnp.asarray(data["log_std"])}
    return params


# region 3. Rollout


class Batch(NamedTuple):
    """Data collected during a rollout, each with leading shape (n_steps, n_envs)."""

    obs: Array
    action: Array
    log_prob: Array
    value: Array
    reward: Array
    done: Array


def rollout(params: dict, states: env.State, key: Array, track: env.Track, cfg: Config) -> tuple:
    """Fly the current policy for ``n_steps`` in every environment."""

    def one_step(states: env.State, key: Array) -> tuple:
        obs = jax.vmap(env.observe_state, (0, None))(states, track)
        noise = jax.random.normal(key, (cfg.n_envs, env.ACTION_DIM))
        action = policy_mean(params["actor"], obs) + jnp.exp(params["log_std"]) * noise
        value = mlp(params["critic"], obs)[..., 0]
        states, reward, done, info = jax.vmap(env.step, (0, 0, None))(states, action, track)
        batch = Batch(obs, action, log_prob(params, obs, action), value, reward, done)
        return states, (batch, info)

    states, (batch, info) = jax.lax.scan(one_step, states, jax.random.split(key, cfg.n_steps))
    last_obs = jax.vmap(env.observe_state, (0, None))(states, track)
    last_value = mlp(params["critic"], last_obs)[..., 0]
    return states, batch, last_value, info


def advantages(batch: Batch, last_value: Array, cfg: Config) -> tuple[Array, Array]:
    """Generalized advantage estimation: how much better each action was than expected."""

    def backward(carry: tuple, x: tuple) -> tuple:
        gae, next_value = carry
        reward, value, done = x
        delta = reward + cfg.gamma * next_value * (1 - done) - value
        gae = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    _, adv = jax.lax.scan(backward, init, (batch.reward, batch.value, batch.done), reverse=True)
    return adv, adv + batch.value


# region 4. PPO update


def ppo_loss(params: dict, batch: Batch, adv: Array, returns: Array, cfg: Config) -> Array:
    """Clipped PPO objective + value regression + entropy bonus."""
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = jnp.exp(log_prob(params, batch.obs, batch.action) - batch.log_prob)
    clipped = jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
    policy_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
    value_loss = jnp.mean((mlp(params["critic"], batch.obs)[..., 0] - returns) ** 2)
    entropy = jnp.sum(params["log_std"])  # Entropy of a Gaussian up to a constant
    return policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy


def make_update(track: env.Track, cfg: Config, optimizer: optax.GradientTransformation) -> callable:
    """Build the compiled function that does one rollout followed by one PPO update."""

    @jax.jit
    def update(
        params: dict, opt_state: optax.OptState, states: env.State, key: Array, time_penalty: Array
    ) -> tuple:
        k_rollout, k_shuffle = jax.random.split(key)
        current = track._replace(time_penalty=time_penalty)
        states, batch, last_value, info = rollout(params, states, k_rollout, current, cfg)
        adv, returns = advantages(batch, last_value, cfg)
        # Flatten (n_steps, n_envs, ...) to (n_samples, ...) for minibatch sampling
        data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (batch, adv, returns))

        def minibatch_step(carry: tuple, idx: Array) -> tuple:
            params, opt_state = carry
            mb = jax.tree.map(lambda x: x[idx], data)
            grads = jax.grad(ppo_loss)(params, *mb, cfg)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), None

        def epoch(carry: tuple, key: Array) -> tuple:
            idx = jax.random.permutation(key, cfg.n_envs * cfg.n_steps)
            return jax.lax.scan(minibatch_step, carry, idx.reshape(cfg.minibatches, -1))

        (params, opt_state), _ = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(k_shuffle, cfg.epochs)
        )
        episodes = jnp.maximum(info["done"].sum(), 1)
        stats = {
            "reward": batch.reward.mean(),  # Average reward per step
            "gates": info["passed"].sum() / episodes,  # Gates passed per episode
            "crash": info["crashed"].sum() / episodes,  # Share of episodes ending in a crash
            "finish": info["finished"].sum() / episodes,  # Share of episodes reaching the last gate
            "std": jnp.exp(params["log_std"]).mean(),  # Exploration noise of the policy
        }
        return params, opt_state, states, stats

    return update


# region 5. Training loop


def train(
    config: str = "level2.toml",
    max_speed: float = env.MAX_SPEED,
    time_penalty: float = env.R_TIME,
    drone_radius: float = env.DRONE_RADIUS,
    max_tilt: float = env.MAX_TILT,
    spin_penalty: float = env.R_SPIN,
    thrust_range: float = env.THRUST_SCALE_RANGE,
    bias_range: float = env.RPY_BIAS_RANGE,
    max_delay: int = env.MAX_DELAY_STEPS,
    init: str | None = None,
    out: str = POLICY_PATH.name,
    **overrides: int | float,
):
    """Train a racing policy and save its weights.

    Args:
        config: Level file in ``config/`` that defines the track.
        max_speed: Speed limit in m/s. Only a ceiling, it does not make the policy hurry.
        time_penalty: Reward lost per step. Higher values give faster laps and more crashes.
            It rises from the default to this value during the first half of training. A policy
            that is hurried from the start never learns to finish the track. With ``init`` the
            policy can already fly, so the value applies from the first step.
        drone_radius: Collision size of the drone in m. Smaller values leave less safety margin
            around gates and poles, which lets the policy fly tighter and faster lines.
        max_tilt: Largest roll and pitch command in rad. More tilt gives sharper turns and a
            more aggressive flight. The value is stored in the weights file.
        spin_penalty: Reward lost per (rad/s)^2 of roll and pitch rate. Lower values allow
            quick turns, at the cost of a less smooth flight.
        thrust_range: The thrust is randomized by up to this fraction, 0.15 means +-15 %.
        bias_range: Roll and pitch are held with a constant random offset of up to this many
            rad, like a real drone that is not perfectly trimmed.
        max_delay: Commands are randomly delayed by 0 to this many control steps of 20 ms.
            Both make the policy tolerate a real drone that differs from the simulation.
        init: File name of saved weights in ``control/e2e/`` to continue from, instead of
            starting with a random network.
        out: File name of the saved weights inside ``control/e2e/``. The controller flies
            ``policy.npz``, so other names can be trained and compared without replacing it.
        overrides: Any field of ``Config``, e.g. ``--total_steps 5e7 --n_envs 4096``.
    """
    cfg = Config()._replace(**{k: type(getattr(Config(), k))(v) for k, v in overrides.items()})
    track = env.load_track(
        config,
        max_speed,
        time_penalty,
        drone_radius,
        max_tilt,
        spin_penalty,
        thrust_range,
        bias_range,
        max_delay,
    )
    path = POLICY_PATH.parent / out
    n_updates = cfg.total_steps // (cfg.n_envs * cfg.n_steps)
    schedule = optax.linear_schedule(
        cfg.learning_rate, 0.0, n_updates * cfg.epochs * cfg.minibatches
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5)
    )

    key, k_params, k_envs = jax.random.split(jax.random.key(cfg.seed), 3)
    params = init_params(k_params, cfg)
    if init is not None:
        params = load_params(POLICY_PATH.parent / init, params, max_tilt)
    opt_state = optimizer.init(params)
    states = jax.vmap(env.reset, (0, None))(jax.random.split(k_envs, cfg.n_envs), track)
    update = make_update(track, cfg, optimizer)

    batch_size = cfg.n_envs * cfg.n_steps
    print(f"Training on {jax.devices()[0]}: {n_updates} updates of {batch_size} steps")
    print(
        f"Speed limit {max_speed} m/s, time penalty {time_penalty}, drone radius {drone_radius} m, "
        f"tilt limit {max_tilt} rad, spin penalty {spin_penalty}, thrust +-{thrust_range:.0%}, "
        f"attitude bias +-{bias_range} rad, delay up to {20 * max_delay} ms, "
        f"start from {init or 'scratch'}, "
        f"saving to {path.name}"
    )
    start = time.time()
    for i in range(1, n_updates + 1):
        key, k_update = jax.random.split(key)
        ramp = 1.0 if init else min(1.0, 2 * i / n_updates)  # 0 -> 1 over the first half
        penalty = env.R_TIME + ramp * (time_penalty - env.R_TIME)
        params, opt_state, states, stats = update(params, opt_state, states, k_update, penalty)
        if i % 10 == 0 or i == n_updates:
            steps = i * batch_size
            print(
                f"{steps / 1e6:7.1f}M steps | {steps / (time.time() - start):8.0f} steps/s | "
                f"reward {stats['reward']:6.3f} | gates {stats['gates']:4.2f} | "
                f"crash {stats['crash']:4.2f} | finish {stats['finish']:4.2f} | "
                f"std {stats['std']:4.2f} | time penalty {penalty:5.3f}"
            )
            save_policy(params, max_tilt, path)
    print(f"Done in {time.time() - start:.0f} s. Policy saved to {path}")


if __name__ == "__main__":
    fire.Fire(train)
