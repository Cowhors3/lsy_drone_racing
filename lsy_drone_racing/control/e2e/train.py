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


def save_policy(params: dict, path: Path = POLICY_PATH):
    """Save the actor weights. The critic is only needed during training."""
    arrays = {}
    for i, (w, b) in enumerate(params["actor"]):
        arrays[f"w{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(path, **arrays)


def load_policy(path: Path = POLICY_PATH) -> list[tuple[Array, Array]]:
    """Load actor weights saved by ``save_policy``."""
    data = np.load(path)
    return [(jnp.asarray(data[f"w{i}"]), jnp.asarray(data[f"b{i}"])) for i in range(len(data) // 2)]


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
    def update(params: dict, opt_state: optax.OptState, states: env.State, key: Array) -> tuple:
        k_rollout, k_shuffle = jax.random.split(key)
        states, batch, last_value, info = rollout(params, states, k_rollout, track, cfg)
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


def train(config: str = "level2.toml", **overrides: int | float):
    """Train a racing policy and save it to ``policy.npz``.

    Args:
        config: Level file in ``config/`` that defines the track.
        overrides: Any field of ``Config``, e.g. ``--total_steps 5e7 --n_envs 4096``.
    """
    cfg = Config()._replace(**{k: type(getattr(Config(), k))(v) for k, v in overrides.items()})
    track = env.load_track(config)
    n_updates = cfg.total_steps // (cfg.n_envs * cfg.n_steps)
    schedule = optax.linear_schedule(
        cfg.learning_rate, 0.0, n_updates * cfg.epochs * cfg.minibatches
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5)
    )

    key, k_params, k_envs = jax.random.split(jax.random.key(cfg.seed), 3)
    params = init_params(k_params, cfg)
    opt_state = optimizer.init(params)
    states = jax.vmap(env.reset, (0, None))(jax.random.split(k_envs, cfg.n_envs), track)
    update = make_update(track, cfg, optimizer)

    batch_size = cfg.n_envs * cfg.n_steps
    print(f"Training on {jax.devices()[0]}: {n_updates} updates of {batch_size} steps")
    start = time.time()
    for i in range(1, n_updates + 1):
        key, k_update = jax.random.split(key)
        params, opt_state, states, stats = update(params, opt_state, states, k_update)
        if i % 10 == 0 or i == n_updates:
            steps = i * batch_size
            print(
                f"{steps / 1e6:7.1f}M steps | {steps / (time.time() - start):8.0f} steps/s | "
                f"reward {stats['reward']:6.3f} | gates {stats['gates']:4.2f} | "
                f"crash {stats['crash']:4.2f} | finish {stats['finish']:4.2f} | "
                f"std {stats['std']:4.2f}"
            )
            save_policy(params)
    print(f"Done in {time.time() - start:.0f} s. Policy saved to {POLICY_PATH}")


if __name__ == "__main__":
    fire.Fire(train)
