"""Geometric model of the race track as seen by the controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

GATE_HALF_WIDTH = 0.36  # Outer half width of the gate frame [m].
GATE_HALF_OPENING = 0.2  # Half width of the gate opening [m].


@dataclass
class Track:
    """Current belief about gates and obstacles plus the gate order.

    Positions are the nominal ones until an object comes into sensor range.

    Attributes:
        gates_pos: Gate centres, shape (n_gates, 3).
        gates_yaw: Gate yaw angles, shape (n_gates,).
        obstacles_pos: Obstacle positions, shape (n_obstacles, 3).
        gates_seen: Flags of gates whose true pose is known.
        obstacles_seen: Flags of obstacles whose true position is known.
        sequence: Gate id for every pass of the race, shape (n_passes,).
        direction: Passing direction (+1 / -1) of every pass, shape (n_passes,).
    """

    gates_pos: NDArray
    gates_yaw: NDArray
    obstacles_pos: NDArray
    gates_seen: NDArray
    obstacles_seen: NDArray
    sequence: NDArray
    direction: NDArray

    @staticmethod
    def from_obs(obs: dict[str, NDArray]) -> Track:
        """Create the track model from an environment observation."""
        n_passes = len(obs["gate_sequence"])
        direction = obs.get("gate_sequence_direction", np.ones(n_passes))
        return Track(
            gates_pos=np.asarray(obs["gates_pos"], float).copy(),
            gates_yaw=R.from_quat(obs["gates_quat"]).as_euler("xyz")[:, 2],
            obstacles_pos=np.asarray(obs["obstacles_pos"], float).copy(),
            gates_seen=np.asarray(obs["gates_visited"], bool).copy(),
            obstacles_seen=np.asarray(obs["obstacles_visited"], bool).copy(),
            sequence=np.asarray(obs["gate_sequence"], int).copy(),
            direction=np.asarray(direction, int).copy(),
        )

    @property
    def n_passes(self) -> int:
        """Number of gate passes of the race."""
        return len(self.sequence)

    def gate_axes(self, gate: int) -> tuple[NDArray, NDArray]:
        """Return the horizontal unit normal and lateral axis of a gate."""
        c, s = np.cos(self.gates_yaw[gate]), np.sin(self.gates_yaw[gate])
        return np.array([c, s, 0.0]), np.array([-s, c, 0.0])

    def pass_frame(self, k: int) -> tuple[NDArray, NDArray, NDArray]:
        """Return centre, flight direction and lateral axis of the k-th gate pass."""
        gate = self.sequence[k]
        normal, lateral = self.gate_axes(gate)
        sign = float(self.direction[k])
        return self.gates_pos[gate], sign * normal, sign * lateral

    def gate_walls(self) -> tuple[NDArray, NDArray]:
        """Return the 2D end points of all gate frames, each of shape (n_gates, 2)."""
        yaw = self.gates_yaw
        lateral = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
        centre = self.gates_pos[:, :2]
        return centre - GATE_HALF_WIDTH * lateral, centre + GATE_HALF_WIDTH * lateral

    def gate_discs(self, offsets: tuple[float, ...]) -> tuple[NDArray, NDArray]:
        """Approximate every gate frame by a row of points along its width.

        Args:
            offsets: Lateral offsets of the points from the gate centre.

        Returns:
            The 2D points of shape (n_gates * len(offsets), 2) and the gate id of every point.
        """
        yaw = self.gates_yaw
        lateral = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
        off = np.asarray(offsets, float)
        pts = self.gates_pos[:, None, :2] + off[None, :, None] * lateral[:, None, :]
        owner = np.repeat(np.arange(len(yaw)), len(off))
        return pts.reshape(-1, 2), owner
