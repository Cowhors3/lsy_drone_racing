"""NumPy version of ``xji_e2e_racer.py``: same policy, same commands, evaluated on the CPU.

The policy network is three small matrix products. Evaluating it with NumPy takes well below a
millisecond per step and involves no GPU transfers, so the step time is short and steady. The JAX
version occasionally exceeds the 20 ms control period when the GPU is busy.

Each control step does three things:

1. Build the policy input from the environment observation (same layout as ``e2e/env.observe``).
2. Evaluate the policy network.
3. Convert the network output to an attitude command [roll, pitch, yaw, thrust].

Requires ``control_mode = "attitude"`` in the level config.

Flies ``e2e/policy.npz`` by default. To try other weights without replacing that file, set the
environment variable ``E2E_POLICY``, e.g. ``E2E_POLICY=policy_fast.npz python scripts/sim.py``.

Every flight on the real drone is recorded to ``logs/<time>_<policy>.npz`` (observations, policy
actions, commands, timing), see ``FlightLog``. Set ``E2E_LOG=1`` to record simulated flights too.
"""

from __future__ import annotations  # Python 3.10 type hints

import atexit
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.e2e import env  # Only for the constants shared with training

if TYPE_CHECKING:
    from numpy.typing import NDArray

POLICY_DIR = Path(__file__).parent / "e2e"
LOG_DIR = Path(__file__).parents[2] / "logs"


def load_policy(path: Path) -> tuple[list[tuple[NDArray, NDArray]], float]:
    """Load the actor layers (``w0, b0, w1, b1, ...``) and the tilt limit they were trained for."""
    data = np.load(path)
    n_layers = sum(key.startswith("w") for key in data.files)
    layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(n_layers)]
    max_tilt = float(data["max_tilt"]) if "max_tilt" in data.files else env.MAX_TILT
    return layers, max_tilt


def observe(obs: dict[str, NDArray[np.floating]], last_actions: NDArray) -> NDArray:
    """Build the policy input, identical to ``e2e/env.observe``."""
    # The next gates to pass. After the last gate, the last one is repeated.
    sequence = np.asarray(obs["gate_sequence"])
    ahead = np.minimum(obs["n_gates_passed"] + np.arange(env.N_GATES_AHEAD), len(sequence) - 1)
    targets = sequence[ahead]
    directions = np.asarray(obs["gate_sequence_direction"])[ahead]

    pos = obs["pos"]
    rot = R.from_quat(obs["quat"]).as_matrix()
    gate_offset = obs["gates_pos"][targets] - pos  # Where the next gates are, relative to the drone
    gate_normal = R.from_quat(obs["gates_quat"][targets]).as_matrix()[:, :, 0] * directions[:, None]
    obstacle_offset = obs["obstacles_pos"][:, :2] - pos[:2]  # Poles are vertical
    parts = [pos, obs["vel"], rot, gate_offset, gate_normal, obstacle_offset, last_actions]
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])


def policy_mean(layers: list[tuple[NDArray, NDArray]], x: NDArray) -> NDArray:
    """The action the policy prefers, in [-1, 1]: an MLP with tanh on every layer."""
    for w, b in layers:
        x = np.tanh(x @ w + b)
    return x


class FlightLog:
    """Records every control step of a flight and writes it to one ``.npz`` file.

    The file holds one array per observation key plus ``time`` (s since the first step),
    ``action`` (policy output in [-1, 1]), ``command`` ([roll, pitch, yaw, thrust]) and
    ``compute_time`` (s), each with one row per control step.
    """

    def __init__(self, policy_file: str):
        """Start an empty log. It is also saved when the program exits, e.g. after Ctrl+C."""
        self._policy = Path(policy_file).stem
        self._rows: list[dict[str, NDArray]] = []
        atexit.register(self.save)

    def add(self, obs: dict[str, NDArray], action: NDArray, command: NDArray, start: float):
        """Store one control step. ``start`` is the ``time.perf_counter`` value at its begin."""
        row = {key: np.array(value) for key, value in obs.items()}
        row |= {"action": action.copy(), "command": command.copy(), "time": np.array(start)}
        row["compute_time"] = np.array(time.perf_counter() - start)
        self._rows.append(row)

    def save(self):
        """Write the recorded steps to ``logs/`` and start a new log."""
        if not self._rows:
            return
        data = {key: np.stack([row[key] for row in self._rows]) for key in self._rows[0]}
        data["time"] = data["time"] - data["time"][0]
        LOG_DIR.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        np.savez(LOG_DIR / f"{stamp}_{self._policy}.npz", **data)
        self._rows = []


class E2ERacerNumpy(Controller):
    """End-to-end neural network racing controller using the attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load the policy.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        params = load_params(config.sim.dynamics, config.sim.drone)
        self._thrust_low = 4 * float(params["thrust_min"])
        self._thrust_high = 4 * float(params["thrust_max"])
        policy_file = os.environ.get("E2E_POLICY", "policy.npz")
        self._layers, self._max_tilt = load_policy(POLICY_DIR / policy_file)
        self._last_actions = np.zeros((env.N_ACTION_HISTORY, env.ACTION_DIM), dtype=np.float32)
        # Real flights (the deploy script has loaded ROS) are always recorded
        record = "rclpy" in sys.modules or os.environ.get("E2E_LOG") == "1"
        self._log = FlightLog(policy_file) if record else None

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
        start = time.perf_counter()
        action = policy_mean(self._layers, observe(obs, self._last_actions))
        self._last_actions = np.roll(self._last_actions, 1, axis=0)
        self._last_actions[0] = action
        # Map the action in [-1, 1] to tilt angles and total thrust, yaw stays at zero
        roll, pitch = action[:2] * self._max_tilt
        thrust = self._thrust_low + (action[2] + 1) / 2 * (self._thrust_high - self._thrust_low)
        command = np.array([roll, pitch, 0.0, thrust], dtype=np.float32)
        if self._log is not None:
            self._log.add(obs, action, command, start)
        return command

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
        """Forget the action history of the previous episode and save its flight log."""
        self._last_actions[:] = 0.0
        if self._log is not None:
            self._log.save()
