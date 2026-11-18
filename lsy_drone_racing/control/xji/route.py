"""Waypoint routes through tracks that are only known at runtime.

A route is the coarse, piecewise linear skeleton of the racing line. For every gate pass it
contains an approach point in front of the gate, the gate centre and an exit point behind it. The
legs between two passes are planned in the horizontal plane with a visibility graph:

* Obstacle poles are discs, gate frames are line segments ("walls") that may not be crossed.
* Candidate nodes are placed on rings around the poles and around both ends of every wall.
* Two nodes are connected if the straight segment between them keeps a clearance to all objects.
* The best node sequence is found with Dijkstra's algorithm on the directed edges of the graph,
  which allows penalising heading changes in addition to the travelled distance. The heading at
  the start (current velocity or previous gate normal) and at the goal (gate normal) are part of
  the cost, so the route leaves and enters the gates in a natural direction.

The smooth racing line is later fitted to this skeleton by the path module.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.xji.track import Track


@dataclass
class RouteParams:
    """Tuning parameters of the route planner. All lengths in metres."""

    approach: float = 0.35  # Distance of the approach point in front of the gate centre.
    exit: float = 0.30  # Distance of the exit point behind the gate centre.
    run_out: float = 0.6  # Straight run-out behind the last gate.
    pole_clearance: float = 0.26  # Required distance of route segments to pole centres.
    wall_clearance: float = 0.22  # Required distance of route segments to gate frames.
    pole_min_clearance: float = 0.14  # Lower bound when the start is already closer than required.
    wall_min_clearance: float = 0.12
    pole_ring: float = 0.33  # Radius of the node ring around poles.
    pole_ring_nodes: int = 6
    wall_ring: float = 0.32  # Radius of the node ring around the ends of gate frames.
    wall_ring_angles: tuple[float, ...] = (-1.2, 0.0, 1.2)
    guide_distance: float = 0.75  # Distance of extra guide nodes in front of / behind gates.
    guide_offsets: tuple[float, ...] = (-0.4, 0.0, 0.4)
    unseen_pole_margin: float = 0.1  # Extra clearance to poles whose true position is unknown.
    unseen_wall_margin: float = 0.1  # Extra clearance to gates whose true pose is unknown.
    turn_weight: float = 0.25  # Cost of heading changes [m / rad].
    start_turn_weight: float = 0.08  # Extra turn cost at the start per (m/s)^2 of speed.
    funnel_width: float = 0.25  # Lateral tolerance to consider the drone lined up with a gate.
    bounds_low: tuple[float, float] = (-2.8, -2.8)  # Area in which route nodes may be placed.
    bounds_high: tuple[float, float] = (2.8, 2.8)


@dataclass
class Route:
    """Waypoints of the racing line.

    Attributes:
        points: Waypoints including the start position, shape (n, 3).
        centres: Index into ``points`` of the gate centre for every planned pass.
        exit: Pass index and waypoint index of the exit point of the gate that has just been
            passed, if the route starts between this gate and its exit point.
    """

    points: NDArray
    centres: dict[int, int] = field(default_factory=dict)
    exit: tuple[int, int] | None = None


def point_segment_distance(p: NDArray, q: NDArray, c: NDArray) -> NDArray:
    """Distance of points ``c`` (m, 2) to segments ``p``-``q`` (..., 2). Returns shape (..., m)."""
    d = q - p
    length2 = np.maximum(np.sum(d * d, axis=-1), 1e-12)
    rel = c - p[..., None, :]
    w = np.clip(np.sum(rel * d[..., None, :], axis=-1) / length2[..., None], 0.0, 1.0)
    closest = p[..., None, :] + w[..., None] * d[..., None, :]
    return np.linalg.norm(c - closest, axis=-1)


def _cross(a: NDArray, b: NDArray) -> NDArray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segment_segment_distance(p: NDArray, q: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """Distance of segments ``p``-``q`` (..., 2) to segments ``a``-``b`` (m, 2). Shape (..., m)."""
    pe, qe = p[..., None, :], q[..., None, :]
    d1 = _cross(b - a, pe - a)
    d2 = _cross(b - a, qe - a)
    d3 = _cross(qe - pe, a - pe)
    d4 = _cross(qe - pe, b - pe)
    intersect = (d1 * d2 < 0) & (d3 * d4 < 0)
    dist = np.minimum(point_segment_distance(p, q, a), point_segment_distance(p, q, b))
    for end in (pe, qe):  # Distance of the end points of p-q to a-b
        ab = b - a
        w = np.clip(
            np.sum((end - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-12), 0, 1
        )
        dist = np.minimum(dist, np.linalg.norm(end - (a + w[..., None] * ab), axis=-1))
    return np.where(intersect, 0.0, dist)


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class RoutePlanner:
    """Plans waypoint routes through a [Track][lsy_drone_racing.control.xji.track.Track]."""

    def __init__(self, params: RouteParams | None = None):
        """Create the planner.

        Args:
            params: Planner parameters. Defaults are used if omitted.
        """
        self.params = params or RouteParams()
        self._graph_track: Track | None = None
        self._graph: tuple[NDArray, NDArray] | None = None
        self._leg_cache: dict[tuple, NDArray] = {}

    # region Route assembly

    def plan(
        self, track: Track, pos: NDArray, vel: NDArray, passed: int, horizon: int, aim: NDArray
    ) -> Route:
        """Plan a route from the current state through the next gates.

        Args:
            track: Current track belief.
            pos: Start position of the route (drone or reference position).
            vel: Velocity at the start, used as initial heading.
            passed: Number of gate passes already completed.
            horizon: Number of upcoming passes to include.
            aim: Target points inside the gates for every pass, shape (n_passes, 3).

        Returns:
            The route starting at ``pos``.
        """
        prm = self.params
        points: list[NDArray] = [np.asarray(pos, float)]
        centres: dict[int, int] = {}
        cur, heading = points[0], self._heading(vel)
        speed = float(np.linalg.norm(vel[:2]))
        exit_point = None
        # Finish the exit of the gate we just went through before turning away from it.
        if passed > 0:
            _, normal, lateral = track.pass_frame(passed - 1)
            rel = cur - aim[passed - 1]
            if -0.05 <= rel @ normal < prm.exit - 0.08 and abs(rel @ lateral) < 0.3:
                cur = aim[passed - 1] + prm.exit * normal
                exit_point = (passed - 1, len(points))
                points.append(cur)
                heading = self._heading(normal)
        last = min(passed + horizon, track.n_passes)
        for k in range(passed, last):
            _, normal, lateral = track.pass_frame(k)
            approach = aim[k] - prm.approach * normal
            rel = aim[k] - cur
            lined_up = (
                k == passed
                and 0.0 < rel @ normal < prm.approach + 0.1
                and abs(rel @ lateral) < prm.funnel_width
            )
            if not lined_up:
                via = self.plan_leg(track, cur[:2], heading, speed, approach[:2], normal[:2])
                points.extend(self._lift(via, cur, approach))
                points.append(approach)
            centres[k] = len(points)
            points.append(aim[k])
            final = k == track.n_passes - 1
            cur = aim[k] + (prm.run_out if final else prm.exit) * normal
            points.append(cur)
            heading, speed = self._heading(normal), 0.0
        return self._dedupe(points, centres, exit_point)

    @staticmethod
    def _heading(direction: NDArray) -> float | None:
        if np.linalg.norm(direction[:2]) < 0.2:
            return None
        return float(np.arctan2(direction[1], direction[0]))

    @staticmethod
    def _lift(via: NDArray, start: NDArray, goal: NDArray) -> list[NDArray]:
        """Assign heights to 2D via points by interpolating along the leg."""
        if len(via) == 0:
            return []
        xy = np.vstack([start[:2], via, goal[:2]])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        z = start[2] + (goal[2] - start[2]) * s[1:-1] / max(s[-1], 1e-9)
        return [np.array([p[0], p[1], zi]) for p, zi in zip(via, z)]

    @staticmethod
    def _dedupe(
        points: list[NDArray],
        centres: dict[int, int],
        exit_point: tuple[int, int] | None,
        tol: float = 0.04,
    ) -> Route:
        """Remove waypoints that nearly coincide with their predecessor, except pinned ones."""
        pinned = set(centres.values()) | ({exit_point[1]} if exit_point else set())
        keep = [0]
        for i in range(1, len(points)):
            if i in pinned:
                if np.linalg.norm(points[i] - points[keep[-1]]) < tol and keep[-1] not in pinned:
                    if keep[-1] != 0:
                        keep.pop()  # Drop the predecessor instead of the pinned waypoint
                keep.append(i)
            elif np.linalg.norm(points[i] - points[keep[-1]]) >= tol:
                keep.append(i)
        remap = {old: new for new, old in enumerate(keep)}
        pts = np.array([points[i] for i in keep])
        if exit_point is not None:
            exit_point = (exit_point[0], remap[exit_point[1]])
        return Route(pts, {k: remap[i] for k, i in centres.items()}, exit_point)

    # region Visibility graph

    def plan_leg(
        self,
        track: Track,
        start: NDArray,
        start_heading: float | None,
        start_speed: float,
        goal: NDArray,
        goal_normal: NDArray,
    ) -> NDArray:
        """Find the via points of a collision-free leg in the horizontal plane.

        Args:
            track: Current track belief.
            start: Start point (2,).
            start_heading: Heading at the start in radians or None if at rest.
            start_speed: Horizontal speed at the start.
            goal: Goal point (2,), usually the approach point of a gate.
            goal_normal: Flight direction through the gate behind the goal (2,).

        Returns:
            The via points between start and goal, shape (n, 2). Empty if the direct connection
            is the best route or no route was found.
        """
        prm = self.params
        static_nodes, static_valid = self._static_graph(track)
        key = (*np.round(start, 4), *np.round(goal, 4), start_heading, round(start_speed, 3))
        if key in self._leg_cache:
            return self._leg_cache[key]
        # Node 0 is the start, node 1 the goal, followed by the track nodes.
        ends = np.vstack([start, goal])
        nodes = np.vstack([ends, static_nodes])
        n = len(nodes)
        valid = np.zeros((n, n), bool)
        valid[2:, 2:] = static_valid
        # Edges from the start / to the goal may begin closer to an object than the clearance.
        r_pole, r_wall = self._clearances(track)
        d_pole, d_wall = self._distances(track, ends[:, None, :], ends[:, None, :])
        # Inside a clearance zone only the minimum clearance is required, otherwise the planner
        # could only retreat from an object the drone is about to fly past.
        r_pole = np.where(d_pole[:, 0] < r_pole, prm.pole_min_clearance, r_pole)
        r_wall = np.where(d_wall[:, 0] < r_wall, prm.wall_min_clearance, r_wall)
        d_pole, d_wall = self._distances(track, ends[:, None, :], nodes[None, :, :])
        r_pole_edge, r_wall_edge = r_pole[:, None, :].copy(), r_wall[:, None, :].copy()
        r_pole_edge[:, :2] = np.minimum(r_pole[0], r_pole[1])  # Direct start - goal edge
        r_wall_edge[:, :2] = np.minimum(r_wall[0], r_wall[1])
        ok = np.all(d_pole >= r_pole_edge, axis=-1) & np.all(d_wall >= r_wall_edge, axis=-1)
        valid[:2, :] = ok
        valid[:, :2] = ok.T
        np.fill_diagonal(valid, False)
        w_start = prm.turn_weight + prm.start_turn_weight * start_speed**2
        goal_heading = float(np.arctan2(goal_normal[1], goal_normal[0]))
        order = self._search(nodes, valid, start_heading, goal_heading, w_start)
        via = nodes[order[1:-1]] if order else np.zeros((0, 2))
        self._leg_cache[key] = via
        return via

    def pole_margins(self, track: Track) -> NDArray:
        """Extra clearance to every pole, non-zero for poles that have not been observed yet."""
        return self.params.unseen_pole_margin * ~track.obstacles_seen

    def _clearances(self, track: Track) -> tuple[NDArray, NDArray]:
        """Required clearance to every pole and every gate frame."""
        prm = self.params
        r_pole = prm.pole_clearance + self.pole_margins(track)
        r_wall = prm.wall_clearance + prm.unseen_wall_margin * ~track.gates_seen
        return r_pole, r_wall

    @staticmethod
    def _distances(track: Track, p: NDArray, q: NDArray) -> tuple[NDArray, NDArray]:
        """Distances of the segments p-q to all poles and all gate frames."""
        p, q = np.broadcast_arrays(p, q)
        wall_a, wall_b = track.gate_walls()
        d_pole = point_segment_distance(p, q, track.obstacles_pos[:, :2])
        return d_pole, segment_segment_distance(p, q, wall_a, wall_b)

    def _static_graph(self, track: Track) -> tuple[NDArray, NDArray]:
        """Nodes around the track objects and their mutual visibility, cached per track belief."""
        if self._graph_track is not track:
            nodes = self._candidate_nodes(track)
            r_pole, r_wall = self._clearances(track)
            d_pole, d_wall = self._distances(track, nodes[:, None, :], nodes[None, :, :])
            valid = np.all(d_pole >= r_pole, axis=-1) & np.all(d_wall >= r_wall, axis=-1)
            self._graph_track, self._graph = track, (nodes, valid)
            self._leg_cache = {}
        return self._graph

    def _candidate_nodes(self, track: Track) -> NDArray:
        """Create the nodes of the visibility graph."""
        prm = self.params
        nodes = []
        ang = np.arange(prm.pole_ring_nodes) * 2 * np.pi / prm.pole_ring_nodes
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        for pole, margin in zip(track.obstacles_pos[:, :2], self.pole_margins(track)):
            nodes.append(pole + (prm.pole_ring + margin) * ring)
        wall_a, wall_b = track.gate_walls()
        for g in range(len(wall_a)):
            normal, lateral = (axis[:2] for axis in track.gate_axes(g))
            radius = prm.wall_ring + prm.unseen_wall_margin * (not track.gates_seen[g])
            for end, sign in ((wall_a[g], -1.0), (wall_b[g], 1.0)):
                a = np.asarray(prm.wall_ring_angles)[:, None]
                nodes.append(end + radius * (np.cos(a) * sign * lateral + np.sin(a) * normal))
            # Guide nodes that allow wide arcs into and out of the gate.
            off = np.asarray(prm.guide_offsets)[:, None]
            for side in (-1.0, 1.0):
                centre = track.gates_pos[g, :2] + side * prm.guide_distance * normal
                nodes.append(centre + off * lateral)
        nodes = np.vstack(nodes)
        inside = np.all((nodes >= prm.bounds_low) & (nodes <= prm.bounds_high), axis=1)
        return nodes[inside]

    def _search(
        self,
        nodes: NDArray,
        valid: NDArray,
        start_heading: float | None,
        goal_heading: float,
        w_start: float,
    ) -> list[int]:
        """Dijkstra search over directed edges with a turn penalty.

        Node 0 is the start, node 1 the goal. Returns the node indices of the best route or an
        empty list if the goal cannot be reached.
        """
        w_turn = self.params.turn_weight
        delta = nodes[None, :, :] - nodes[:, None, :]
        length = np.linalg.norm(delta, axis=-1)
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        neighbours = [np.flatnonzero(valid[i]).tolist() for i in range(len(nodes))]
        length_l, angle_l = length.tolist(), angle.tolist()
        heap: list[tuple[float, int, int]] = []
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        done: set[tuple[int, int]] = set()
        for j in neighbours[0]:
            turn = 0.0 if start_heading is None else _angle_diff(angle_l[0][j], start_heading)
            heapq.heappush(heap, (length_l[0][j] + w_start * turn, 0, j))
            parent[(0, j)] = None
        best_cost: dict[tuple[int, int], float] = {(0, j): c for c, _, j in heap}
        while heap:
            cost, i, j = heapq.heappop(heap)
            if (i, j) in done:
                continue
            done.add((i, j))
            if j == -1:  # Terminal state: reconstruct the route
                order, state = [1], parent[(i, j)]
                while state is not None:
                    order.append(state[0])
                    state = parent[state]
                return order[::-1]
            if j == 1:
                total = cost + w_turn * _angle_diff(angle_l[i][j], goal_heading)
                if total < best_cost.get((1, -1), np.inf):
                    best_cost[(1, -1)] = total
                    parent[(1, -1)] = (i, j)
                    heapq.heappush(heap, (total, 1, -1))
                continue
            for m in neighbours[j]:
                if m == i:
                    continue
                new = cost + length_l[j][m] + w_turn * _angle_diff(angle_l[i][j], angle_l[j][m])
                if new < best_cost.get((j, m), np.inf):
                    best_cost[(j, m)] = new
                    parent[(j, m)] = (i, j)
                    heapq.heappush(heap, (new, j, m))
        return []
