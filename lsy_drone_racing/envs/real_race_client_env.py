"""Client environment for multi-drone racing.

The RealMultiDroneRaceEnvClient operates as a client in a host-client system:
- Manages a single drone's state and actions
- Sends actions and state updates to the host
- Receives coordination messages from the host via ROS2
- Handles local observation and gate tracking
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np
from crazyflow.control import load_params as load_control_params
from crazyflow.dynamics import load_params as load_dynamics_params
from drone_estimators.ros_nodes.ros2_connector import ROSConnector
from drone_racing_msgs.msg import RealClientAction, RealHostState  # type: ignore[import-untyped]
from gymnasium import Env
from std_msgs.msg import String

from lsy_drone_racing.envs.real_race_env import EnvData
from lsy_drone_racing.envs.utils import gate_passed, load_gate_order, load_track
from lsy_drone_racing.utils.ros import track_poses
from lsy_drone_racing.utils.ros_race_comm import LATCHED, RaceCommNode

if TYPE_CHECKING:
    from crazyflow.dynamics import Dynamics
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RealMultiDroneRaceEnvClient(Env):
    """Client environment for multi-drone racing.

    We run one client for each drone, and coordinate between clients with a race host. Clients and
    host communicate via ROS2. Clients compute observations and gate tracking locally, and forward
    actions to the host which sends them to the real drones.

    Observation space:
        A dictionary containing the state of all drones in the race, mirroring
        :class:`lsy_drone_racing.envs.multi_drone_race.MultiDroneRaceEnv`.

    Action space:
        A single action vector for the drone identified by ``rank``.

    Note:
        rclpy must be initialized before creating this environment.
    """

    def __init__(
        self,
        drones: list[dict[str, int]],
        rank: int,
        freq: int,
        track: ConfigDict,
        dynamics: Dynamics,
        sensor_range: float = 0.5,
        control_mode: Literal["state", "attitude"] = "state",
    ):
        """Initialize the multi-drone race client.

        Args:
            drones: List of all drones in the race, each with ``id``, ``channel``, and ``drone``
                keys.
            rank: Index of this drone among all drones in the race.
            freq: Control frequency in Hz.
            track: Track configuration (see :func:`~lsy_drone_racing.envs.utils.load_track`).
            dynamics: Dynamics model used to load the drone's physical parameters.
            sensor_range: Distance in metres at which gate/obstacle true poses are revealed.
            control_mode: Control mode of the drone.
        """
        self.n_drones = len(drones)
        self.rank = rank
        self.freq = freq
        self.sensor_range = sensor_range
        self.control_mode = control_mode
        self.drone_names = [f"cf{drone['id']}" for drone in drones]
        self.drone_name = self.drone_names[rank]
        drone_model = drones[rank]["drone"]
        self.drone_parameters: dict = load_dynamics_params(dynamics, drone_model)
        # PWM limits are firmware control parameters, separate from the dynamics parameters.
        control_params = load_control_params("mellinger", drone_model)["core"]
        self.drone_parameters.update(pwm_min=control_params["pwm_min"])
        self.drone_parameters.update(pwm_max=control_params["pwm_max"])

        self.gates, self.obstacles, self.drones_track = load_track(track)
        self.n_gates = len(self.gates.pos)
        self.gate_sequence, self.gate_sequence_direction = load_gate_order(track, self.n_gates)
        self.n_obstacles = len(self.obstacles.pos)
        self.pos_limit_low = np.array(track.safety_limits["pos_limit_low"])
        self.pos_limit_high = np.array(track.safety_limits["pos_limit_high"])

        self.device = jax.devices("cpu")[0]
        self.data = EnvData.create(self.n_drones, self.n_gates, self.n_obstacles)

        self._ros_connector = ROSConnector(
            estimator_names=self.drone_names,
            cmd_topic=f"/drones/{self.drone_name}/command",
            timeout=10.0,
        )
        self._client_action_pub: Any = None
        self._comm = self._init_comm()

        self._race_started = threading.Event()
        self._race_start_time = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset the environment and open the ROS connections.

        Args:
            seed: Unused in real environments.
            options: Deploy options to determine whether to load real track object poses.

        Returns:
            Initial observation and info dictionaries.

        Raises:
            TimeoutError: If the state of a drone is not received within 10 seconds.
        """
        if options and options.get("real_track_objects", False):
            self.gates.pos, self.gates.quat, self.obstacles.pos = track_poses(
                self.n_gates, self.n_obstacles
            )
        current_pos, _, _, _ = self._all_drone_states()
        self.data.reset(current_pos)
        logger.debug("Environment reset complete")
        return self.obs(), self.info()

    def wait_for_race_start(self, timeout: float = 60.0):
        """Wait for the race to start.

        The client has to wait until all clients have joined the race and the host is ready.

        Args:
            timeout: Maximum wait time in seconds.

        Raises:
            TimeoutError: On exceeding the timeout.
        """
        pub = self._comm.node.create_publisher(String, "/race_registration", LATCHED)
        pub.publish(String(data=f"register_{self.rank}"))
        if not self._race_started.wait(timeout=timeout):
            raise TimeoutError("Timeout waiting for race start. Is the host running and reachable?")
        assert self.race_time, "Race time should be available after race has started"
        logger.info("Race starts!")

    def step(self, action: NDArray) -> tuple[dict, float, bool, bool, dict]:
        """Perform a control step: update gate tracking, check bounds, and send the action.

        Args:
            action: Control action for this drone.

        Returns:
            Observation, reward (always 0.0), terminated, truncated (always False), info.
        """
        drone_pos, _, _, _ = self._all_drone_states()

        dpos = drone_pos[:, None, :2] - self.gates.pos[None, :, :2]
        self.data.gates_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range
        dpos = drone_pos[:, None, :2] - self.obstacles.pos[None, :, :2]
        self.data.obstacles_visited |= np.linalg.norm(dpos, axis=-1) < self.sensor_range

        # Allow for different gate ordering
        gate_id = self.gate_sequence[self.data.n_gates_passed[self.rank]]
        gate_reverse = self.gate_sequence_direction[self.data.n_gates_passed[self.rank]] < 0
        gate_pos = self.gates.pos[gate_id]
        gate_quat = self.gates.quat[gate_id]

        with jax.default_device(self.device):
            passed = gate_passed(
                drone_pos, self.data.last_drone_pos, gate_pos, gate_quat, gate_reverse, (0.45, 0.45)
            )
        self.data.n_gates_passed = self.data.n_gates_passed + np.asarray(passed)
        self.data.last_drone_pos[...] = drone_pos
        self.data.taken_off |= drone_pos[self.rank, 2] > 0.1

        terminated = bool(self.data.n_gates_passed[self.rank] >= len(self.gate_sequence))

        pos = drone_pos[self.rank]
        if not np.all((pos >= self.pos_limit_low) & (pos <= self.pos_limit_high)):
            logger.warning("Drone exceeded safety bounds")
            terminated = True

        self._send_action_update(action, terminated)  # TODO: who sends the estimator updates?

        return self.obs(), 0.0, terminated, False, self.info()

    def obs(self) -> dict[str, NDArray]:
        """Return the current observation dictionary."""
        mask = self.data.gates_visited[..., None]
        gates_pos = np.where(mask, self.gates.pos, self.gates.nominal_pos).astype(np.float32)
        gates_quat = np.where(mask, self.gates.quat, self.gates.nominal_quat).astype(np.float32)
        mask = self.data.obstacles_visited[..., None]
        obstacles_pos = np.where(mask, self.obstacles.pos, self.obstacles.nominal_pos).astype(
            np.float32
        )
        drone_pos, drone_quat, drone_vel, drone_ang_vel = self._all_drone_states()
        return {
            "pos": drone_pos,
            "quat": drone_quat,
            "vel": drone_vel,
            "ang_vel": drone_ang_vel,
            "n_gates_passed": self.data.n_gates_passed,
            "gate_sequence": np.broadcast_to(
                self.gate_sequence, (self.n_drones, len(self.gate_sequence))
            ),
            "gate_sequence_direction": np.broadcast_to(
                self.gate_sequence_direction, (self.n_drones, len(self.gate_sequence))
            ),
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
            "gates_visited": self.data.gates_visited,
            "obstacles_pos": obstacles_pos,
            "obstacles_visited": self.data.obstacles_visited,
        }

    def info(self) -> dict:
        """Return the info dictionary."""
        n_gates_passed = int(self.data.n_gates_passed[self.rank])
        return {
            "rank": self.rank,
            "n_gates_passed": n_gates_passed,
            "finished_track": n_gates_passed >= len(self.gate_sequence),
        }

    def close(self):
        """Send a final stop message and close all ROS connections."""
        logger.info("Closing environment...")
        if self._race_started.is_set():
            pub = self._comm.node.create_publisher(String, "/race_registration", LATCHED)
            pub.publish(String(data=f"deregister_{self.rank}"))
        self._comm.close()
        self._ros_connector.close()
        logger.debug("Environment closed")

    def _send_action_update(self, action: NDArray, stopped: bool):
        """Publish a :class:`RealClientAction` to the host.

        The timestamp is taken from the host clock, so the host can compare it with its own time.

        Args:
            action: Current control action.
            stopped: Whether this client has finished or crashed.
        """
        assert self._race_started.is_set(), "Cannot send action before race has started"
        assert isinstance(action, np.ndarray) and action.ndim == 1, "Action must be a 1D np.ndarray"
        msg = RealClientAction()
        msg.rank = self.rank
        msg.action = action.tolist()
        msg.elapsed_time = time.time() - self._race_start_time
        msg.timestamp = self._host_time()
        self._client_action_pub.publish(msg)

    def _host_time(self) -> float:
        """Read the host clock in seconds.

        Returns 0.0 if no time has arrived yet.
        """
        return self._comm.node.get_clock().now().nanoseconds / 1e9

    def _init_comm(self):
        """Set up the ROS2 communication node with all publishers and subscribers."""
        self._comm = RaceCommNode(f"lsy_race_client_{self.rank}", use_sim_time=True)
        node = self._comm.node

        def on_host_state(msg: RealHostState):
            latency_ms = (time.time() - msg.timestamp) * 1000
            if msg.race_started and not self._race_started.is_set():
                self._race_started.set()
                self._race_start_time = time.time() - msg.elapsed_time
                logger.debug(f"Race started (latency: {latency_ms:.2f}ms)")

        self._sub = node.create_subscription(
            RealHostState, "lsy_drone_racing/host_state", on_host_state, 10
        )
        self._client_action_pub = node.create_publisher(
            RealClientAction, f"lsy_drone_racing/client/drone_{self.rank}/action", 10
        )
        logger.debug("ROS2 communication initialized")

    def _all_drone_states(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Read positions, quaternions, velocities, and angular velocities for all drones.

        Returns:
            Tuple of ``(pos, quat, vel, ang_vel)``, each of shape ``(n_drones, ...)``.
        """
        pos = np.concat([self._ros_connector.pos[i] for i in self.drone_names], axis=0)
        quat = np.concat([self._ros_connector.quat[i] for i in self.drone_names], axis=0)
        vel = np.concat([self._ros_connector.vel[i] for i in self.drone_names], axis=0)
        ang_vel = np.concat([self._ros_connector.ang_vel[i] for i in self.drone_names], axis=0)
        return pos, quat, vel, ang_vel
