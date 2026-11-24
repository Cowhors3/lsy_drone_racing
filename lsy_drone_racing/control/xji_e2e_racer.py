"""Controller that flies the end-to-end racing policy trained with ``control/e2e/train.py``.

Each control step does three things:

1. Build the policy input from the environment observation (same function as in training).
2. Evaluate the policy network.
3. Convert the network output to an attitude command [roll, pitch, yaw, thrust].

Requires ``control_mode = "attitude"`` in the level config.

Flies ``e2e/policy.npz`` by default. To try other weights without replacing that file, set the
environment variable ``E2E_POLICY``, e.g. ``E2E_POLICY=policy_fast.npz python scripts/sim.py``.
"""

from __future__ import annotations  # Python 3.10 type hints

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.e2e import env
from lsy_drone_racing.control.e2e.train import POLICY_PATH, load_policy, policy_mean

if TYPE_CHECKING:
    from numpy.typing import NDArray


class E2ERacer(Controller):
    """End-to-end neural network racing controller using the attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy and compile it.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        params = load_params(config.sim.dynamics, config.sim.drone)
        self._thrust_low = 4 * params["thrust_min"]
        self._thrust_high = 4 * params["thrust_max"]
        self._last_actions = np.zeros((env.N_ACTION_HISTORY, env.ACTION_DIM), dtype=np.float32)

        policy_file = os.environ.get("E2E_POLICY", POLICY_PATH.name)
        actor, self._max_tilt = load_policy(POLICY_PATH.with_name(policy_file))

        @jax.jit
        def act(*observation: jax.Array) -> jax.Array:
            return policy_mean(actor, env.observe(*observation))

        self._act = act
        self.compute_control(obs)  # Warm-up call, compiles the policy before the race starts
        self._last_actions[:] = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude command.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The command [roll, pitch, yaw, thrust] as a numpy array.
        """
        # The next gates to pass. After the last gate, the last one is repeated.
        sequence = np.asarray(obs["gate_sequence"])
        ahead = np.minimum(obs["n_gates_passed"] + np.arange(env.N_GATES_AHEAD), len(sequence) - 1)
        action = self._act(
            obs["pos"],
            obs["vel"],
            R.from_quat(obs["quat"]).as_matrix(),
            obs["gates_pos"],
            obs["gates_quat"],
            obs["obstacles_pos"],
            sequence[ahead],
            np.asarray(obs["gate_sequence_direction"])[ahead],
            self._last_actions,
        )
        self._last_actions = np.roll(self._last_actions, 1, axis=0)
        self._last_actions[0] = action
        command = env.action_to_command(
            jnp.asarray(action), self._thrust_low, self._thrust_high, self._max_tilt
        )
        return np.asarray(command, dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """The policy keeps flying until the environment ends the episode."""
        return False

    def episode_callback(self):
        """Forget the action history of the previous episode."""
        self._last_actions[:] = 0.0
