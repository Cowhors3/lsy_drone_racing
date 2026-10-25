"""Spline-based racing controller with gate-aware waypoints and re-planning.

Strategy (model-based, 13D "state" interface):

1. For every remaining gate pass in the configured gate order we place three waypoints along the
   gate's passing axis: an approach point in front of the gate, the gate center, and an exit point
   behind the gate.
2. The trajectory is a chain of cubic-spline "legs" from gate center to gate center. At every gate
   center the velocity is *prescribed* along the gate normal, so the drone crosses each gate
   perpendicular and through the middle of the opening. Inside a leg the spline is C2.
3. Time stamps come from an acceleration-limited speed profile that slows down in sharp turns;
   a time-scaling pass stretches segments whose reference acceleration or speed gets too high.
4. Waypoints are pushed away from obstacle poles, and extra via-points are inserted wherever the
   sampled spline would hit an obstacle pole, a gate stand, or a gate frame.
5. Whenever the observed gate/obstacle positions change (an object entered the sensor range and its
   true position was revealed), the trajectory is re-planned from the current reference state, so
   the setpoints stay continuous.
6. A slow outer-loop integrator on the position error removes the steady-state offset of the
   onboard controller (e.g., caused by the randomized mass).

The position, velocity and acceleration of the trajectory are sent as the setpoint; the onboard
(Mellinger-type) controller does the tracking.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class SplineRacer(Controller):
    """Gate-aware spline trajectory controller with re-planning."""

    # ---- Tunable parameters -------------------------------------------------------------------
    V_CRUISE = 1.0  # [m/s] Maximum path speed. Increase once the success rate is high.
    V_GATE = 0.9  # [m/s] Speed when crossing a gate center.
    V_GATE_TURN = 0.6  # [m/s] Speed through a gate that is followed by a U-turn.
    A_RAMP = 2.0  # [m/s^2] Acceleration used to build the speed profile.
    A_MAX = 4.0  # [m/s^2] Reference acceleration limit enforced by time scaling.
    V_MAX_FACTOR = 1.25  # Reference speed may exceed V_CRUISE by at most this factor.
    V_TURN_MIN = 0.35  # [m/s] Lower bound of the speed limit in sharp turns.
    D_PRE = 0.25  # [m] Distance of the approach waypoint in front of a gate.
    D_POST = 0.22  # [m] Distance of the exit waypoint behind a gate.
    D_EXTRA = 0.35  # [m] Extra point after the very last gate so we do not slow down there.
    R_SAFE = 0.22  # [m] Required horizontal clearance to obstacle poles.
    R_STAND = 0.20  # [m] Required horizontal clearance to the stand below a gate.
    GATE_OPEN = 0.115  # [m] Max. center offset that still clears the frame (0.2 - drone radius).
    GATE_OUTER = 0.55  # [m] Half-width of the keep-out square around the gate frame.
    GATE_SLAB = 0.11  # [m] Half-thickness of the keep-out slab around the gate plane.
    UNC_MARGIN = 0.10  # [m] Extra clearance for objects whose true position is not yet revealed.
    D_POST_TURN = 0.16  # [m] Exit point distance if the path has to turn around behind the gate.
    D_TURN = 0.62  # [m] Lateral offset of the turn-around point after a gate (U-turn behind it).
    TAKEOFF_Z = 0.25  # [m] Height of the first waypoint above the start (None to disable).
    REPLAN_POS_TOL = 0.02  # [m] Observed object position change that triggers a re-plan.
    MAX_TRACK_ERR = 0.6  # [m] Re-plan from the measured state if the drone is this far off.
    START_IGNORE = 0.12  # [m] Collisions this close to the plan start cannot be fixed by planning.
    KI = np.array([0.5, 0.5, 1.5])  # [1/s] Outer-loop integral gain on the position error.
    I_LIM = np.array([0.08, 0.08, 0.12])  # [m] Anti-windup bound of the integral correction.
    GRID_RES = 0.07  # [m] Voxel size of the A* router.
    A_INFLATE = 0.04  # [m] Extra clearance of the router around keep-out zones.
    A_WEIGHT = 1.4  # Weight of the A* heuristic (>1 trades optimality for speed).
    TIMEOUT = 25.0  # [s] Abort the episode if it takes longer than this.
    XY_LIM = (2.3, 1.3)  # [m] Keep waypoints inside the safety limits.
    Z_LIM = (0.1, 1.8)  # [m]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and plan the first trajectory.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._gate_seq = np.asarray(obs["gate_sequence"], dtype=int)
        self._gate_dir = np.asarray(obs["gate_sequence_direction"], dtype=int)
        self._quat = R.from_euler("z", 0.0).as_quat()

        self._tick = 0
        self._finished = False
        self._last_replan_tick = 0
        self._t_start = 0.0  # Time (s) at which the current trajectory started.
        self._takeoff_done = False
        self._int = np.zeros(3)  # Integral correction added to the position setpoint.
        self.ref_pos = np.array(obs["pos"], dtype=float)  # Last reference (for diagnostics).
        self.n_replans = 0
        self.plan_log: list[tuple[int, bool, int]] = []  # (tick, unresolved collision, n waypoints)

        self._plan(obs, np.array(obs["pos"], dtype=float), np.zeros(3))

    # ------------------------------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _gate_axis(quat: NDArray, direction: int) -> NDArray:
        """Passing direction of a gate in the world frame (local +x, flipped for reverse)."""
        return R.from_quat(quat).apply([1.0, 0.0, 0.0]) * float(direction)

    def _waypoints(
        self, obs: dict[str, NDArray], start: NDArray
    ) -> tuple[list[NDArray], list[bool], list[NDArray | None]]:
        """Build the waypoint list for all remaining gate passes.

        Returns:
            The waypoints, a flag per waypoint telling if it may be moved for obstacle avoidance
            (gate centers and approach points keep their position/direction), and the prescribed
            passing direction for anchors (None otherwise).
        """
        n_passed = int(obs["n_gates_passed"])
        gates_pos, gates_quat = np.asarray(obs["gates_pos"]), np.asarray(obs["gates_quat"])
        n_seq = len(self._gate_seq)
        self._slow_centers: list[NDArray] = []
        pts: list[NDArray] = [np.array(start, dtype=float)]
        movable: list[bool] = [False]
        normals: list[NDArray | None] = [None]

        def add(p: NDArray, mov: bool, normal: NDArray | None = None):
            pts.append(np.array(p, dtype=float))
            movable.append(mov)
            normals.append(normal)

        def gate(i: int) -> tuple[NDArray, NDArray, NDArray]:
            """Center, passing direction and lateral axis (world frame) of gate pass ``i``."""
            q = gates_quat[self._gate_seq[i]]
            return (
                gates_pos[self._gate_seq[i]],
                self._gate_axis(q, self._gate_dir[i]),
                R.from_quat(q).apply([0.0, 1.0, 0.0]),
            )

        def next_target(i: int) -> NDArray | None:
            """First waypoint after the exit of gate pass ``i`` (approach point of the next)."""
            if i + 1 >= n_seq:
                return None
            c_next, n_next, _ = gate(i + 1)
            return c_next - n_next * self.D_PRE

        dir_cache: dict[int, tuple[NDArray, float]] = {}

        def axis_dir(i: int) -> tuple[NDArray, float]:
            """Crossing direction for gate pass ``i``.

            The gate normal, or the closest rotated version (about the vertical) whose approach
            segment is free of obstacles.
            """
            if i not in dir_cache:
                c, n, _ = gate(i)
                dir_cache[i] = (n, 0.0)
                for deg in (0, 15, -15, 30, -30, 45, -45, 60, -60):
                    a = np.deg2rad(deg)
                    rot = np.array(
                        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
                    )
                    u = rot @ n
                    seg = c[None] - u[None] * np.linspace(0.1, self.D_PRE, 6)[:, None]
                    if not self._blocked(seg, obs, 0.0).any():
                        dir_cache[i] = (u, float(deg))
                        break
                else:
                    dir_cache[i] = (n, 999.0)  # nothing free: keep the normal, enter close
            return dir_cache[i]

        def needs_turnaround(i: int) -> bool:
            """True if the next gate pass requires re-crossing the plane of pass ``i``."""
            target = next_target(i)
            if target is None:
                return False
            c, n, _ = gate(i)
            return bool(np.dot(target - c, n) < 0.05)

        def add_exit(i: int):
            """Add exit point (and a turn-around detour if the next gate is behind this one)."""
            c, n, y_axis = gate(i)
            u, _ = axis_dir(i)
            if not needs_turnaround(i):
                add(c + u * self.D_POST, True)
                return
            add(c + u * self.D_POST_TURN, True)  # The A* router finds the way around.

        # Take off vertically first if we start on the ground.
        if (
            self.TAKEOFF_Z is not None
            and not self._takeoff_done
            and n_passed == 0
            and start[2] < self.TAKEOFF_Z - 0.05
        ):
            add([start[0], start[1], self.TAKEOFF_Z], False)
        self._takeoff_done = True

        # Keep the exit of the gate we just passed until the drone is clear of it.
        if n_passed > 0:
            c, n, _ = gate(n_passed - 1)
            if np.linalg.norm(start - c) < 0.9 and -0.05 < np.dot(start - c, n) < 0.7 * self.D_POST:
                add_exit(n_passed - 1)

        for i in range(n_passed, n_seq):
            c, n, _ = gate(i)
            u, deg = axis_dir(i)
            if i > n_passed or float(np.dot(start - c, u)) < -(self.D_PRE + 0.04):
                if deg == 999.0:  # Something sits on the approach axis: enter close to the gate.
                    add(c - n * 0.12, False)
                else:
                    add(c - u * self.D_PRE, deg == 0.0, u)
            add(c, False, u)
            if needs_turnaround(i):
                self._slow_centers.append(c.copy())
            if i == n_seq - 1:
                add(c + u * self.D_POST, True)
                add(c + u * (self.D_POST + self.D_EXTRA), True)
            else:
                add_exit(i)

        for k, p in enumerate(pts):  # Stay inside the arena
            p[0] = np.clip(p[0], -self.XY_LIM[0], self.XY_LIM[0])
            p[1] = np.clip(p[1], -self.XY_LIM[1], self.XY_LIM[1])
            if k > 0:
                p[2] = np.clip(p[2], self.Z_LIM[0], self.Z_LIM[1])
        return pts, movable, normals

    def _build_trajectory(
        self,
        pts: list[NDArray],
        normals: list[NDArray | None],
        v0: NDArray,
        scale_time: bool = False,
    ) -> tuple[NDArray, CubicHermiteSpline]:
        """Build the chain of spline legs through the waypoints.

        Args:
            pts: Waypoints, the first one is the plan start.
            normals: Passing direction at gate centers (None for other waypoints). Leg boundaries
                are placed at the gate centers where the velocity is prescribed along the normal.
            v0: Velocity at the start.
            scale_time: Stretch segments whose acceleration/speed exceeds the limits.

        Returns:
            The knot times and a piecewise polynomial (position, derivatives via ``nu``).
        """
        pts_arr = np.asarray(pts)
        n_pts = len(pts_arr)
        diff = np.diff(pts_arr, axis=0)
        seg = np.maximum(np.linalg.norm(diff, axis=1), 1e-3)
        dirs = diff / seg[:, None]

        # --- Speed profile (limits -> forward/backward acceleration passes) ---
        v_lim = np.full(n_pts, self.V_CRUISE)
        for k in range(1, n_pts - 1):
            cos_turn = float(np.clip(np.dot(dirs[k - 1], dirs[k]), -1.0, 1.0))
            half_turn_cos = np.sqrt(max(0.0, 0.5 * (1.0 + cos_turn)))  # = cos(turn / 2)
            v_lim[k] = self.V_CRUISE * max(half_turn_cos, self.V_TURN_MIN / self.V_CRUISE)
        for k in range(n_pts):
            if normals[k] is not None:
                slow = any(np.linalg.norm(pts_arr[k] - c) < 1e-6 for c in self._slow_centers)
                v_lim[k] = min(v_lim[k], self.V_GATE_TURN if slow else self.V_GATE)
        v = v_lim.copy()
        v[0] = min(float(np.linalg.norm(v0)), self.V_CRUISE)
        for k in range(len(seg)):
            v[k + 1] = min(v[k + 1], np.sqrt(v[k] ** 2 + 2 * self.A_RAMP * seg[k]))
        for k in range(len(seg) - 1, -1, -1):
            v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * self.A_RAMP * seg[k]))
        v_avg = np.maximum(0.5 * (v[:-1] + v[1:]), 0.25)
        dt = seg / v_avg

        # --- Legs between gate centers ---
        bounds = sorted({0, n_pts - 1} | {k for k in range(n_pts) if normals[k] is not None})
        t_arr = np.concatenate(([0.0], np.cumsum(dt)))
        poly = self._chain_legs(pts_arr, t_arr, normals, self._anchor_speed(seg, dt), v0, bounds)
        for _ in range(10 if scale_time else 0):
            ts = np.linspace(0.0, t_arr[-1], max(80, int(t_arr[-1] * 80)))
            acc = np.linalg.norm(poly(ts, 2), axis=1)
            vel = np.linalg.norm(poly(ts, 1), axis=1)
            idx = np.clip(np.searchsorted(t_arr, ts, side="right") - 1, 0, len(seg) - 1)
            scale = np.ones(len(seg))
            for k in range(len(seg)):
                m = idx == k
                if m.any():
                    r_a = np.sqrt(acc[m].max() / self.A_MAX)
                    r_v = vel[m].max() / (self.V_MAX_FACTOR * self.V_CRUISE)
                    scale[k] = max(1.0, r_a, r_v)
            if np.all(scale < 1.02):
                break
            dt = dt * scale
            t_arr = np.concatenate(([0.0], np.cumsum(dt)))
            poly = self._chain_legs(
                pts_arr, t_arr, normals, self._anchor_speed(seg, dt), v0, bounds
            )
        return t_arr, poly

    @staticmethod
    def _anchor_speed(seg: NDArray, dt: NDArray) -> NDArray:
        """Speed to prescribe at each waypoint: the chord speed over its adjacent segments.

        Keeping the prescribed tangent magnitude consistent with the segment durations avoids
        spline overshoot when segments get stretched by the time scaling.
        """
        length = np.concatenate(([0.0], seg)) + np.concatenate((seg, [0.0]))
        time = np.concatenate(([0.0], dt)) + np.concatenate((dt, [0.0]))
        return length / np.maximum(time, 1e-6)

    @staticmethod
    def _chain_legs(
        pts: NDArray,
        t: NDArray,
        normals: list[NDArray | None],
        v: NDArray,
        v0: NDArray,
        bounds: list[int],
    ) -> CubicHermiteSpline:
        """C1 cubic Hermite spline through the waypoints (local tangents, no global overshoot).

        Tangents are the chord speed times the bisector of the adjacent segment directions
        (magnitude shrinks in sharp turns), or the prescribed gate normal at anchors.
        """
        n = len(pts)
        seg = pts[1:] - pts[:-1]
        unit = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
        tang = np.zeros((n, 3))
        tang[0] = v0
        for k in range(1, n - 1):
            if normals[k] is not None:
                tang[k] = v[k] * normals[k]
            else:
                tang[k] = 0.5 * v[k] * (unit[k - 1] + unit[k])
        return CubicHermiteSpline(t, pts, tang)

    def _first_violation(
        self, samples: NDArray, obs: dict[str, NDArray]
    ) -> tuple[int, NDArray] | None:
        """Find the earliest sample that collides with the track and a via-point that avoids it.

        Checks obstacle poles, gate stands (vertical posts below the gates), and the gate frames
        (only crossing the gate plane through the opening is allowed).

        Returns:
            The sample index and a world-frame via-point, or None if the path is collision free.
        """
        hits: list[tuple[int, NDArray]] = []
        # Samples right at the plan start are where the drone already is: planning cannot fix them.
        active = np.linalg.norm(samples - samples[0], axis=1) > self.START_IGNORE

        def radial_via(center_xy: NDArray, i: int, radius: float) -> NDArray:
            d = samples[i, :2] - center_xy
            dist = float(np.linalg.norm(d))
            u = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
            return np.array([*(center_xy + u * (radius + 0.06)), samples[i, 2]])

        obst_seen = np.asarray(obs["obstacles_visited"], dtype=bool)
        for o, seen in zip(np.asarray(obs["obstacles_pos"])[:, :2], obst_seen):  # Obstacle poles
            r = self.R_SAFE + (0.0 if seen else self.UNC_MARGIN)
            bad = np.where(active & (np.linalg.norm(samples[:, :2] - o, axis=1) < r - 0.01))[0]
            if len(bad):
                hits.append((int(bad[0]), radial_via(o, int(bad[len(bad) // 2]), r)))

        gates_pos, gates_quat = np.asarray(obs["gates_pos"]), np.asarray(obs["gates_quat"])
        gate_seen = np.asarray(obs["gates_visited"], dtype=bool)
        for c, q, seen in zip(gates_pos, gates_quat, gate_seen):
            unc = 0.0 if seen else self.UNC_MARGIN
            # Stand below the gate
            below = samples[:, 2] < c[2] - 0.25
            dxy = np.linalg.norm(samples[:, :2] - c[:2], axis=1)
            bad = np.where(active & below & (dxy < self.R_STAND + unc))[0]
            if len(bad):
                hits.append(
                    (int(bad[0]), radial_via(c[:2], int(bad[len(bad) // 2]), self.R_STAND + unc))
                )
            # Frame: samples in the slab around the gate plane but outside the opening
            rot = R.from_quat(q)
            local = rot.inv().apply(samples - c)
            r_inf = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
            bad = np.where(
                active
                & (np.abs(local[:, 0]) < self.GATE_SLAB)
                & (r_inf > self.GATE_OPEN)
                & (r_inf < self.GATE_OUTER + unc)
            )[0]
            if len(bad):
                i = int(bad[len(bad) // 2])
                _, y, z = local[i]
                out = self.GATE_OUTER + unc + 0.06
                options = [  # (displacement, new local y, new local z): sideways or up, never down
                    (out - abs(y), np.sign(y) * out if y != 0 else out, z),
                    (out - z, y, out),
                ]
                _, y_new, z_new = min(options, key=lambda o: o[0])
                hits.append((int(bad[0]), c + rot.apply([local[i, 0], y_new, z_new])))

        if not hits:
            return None
        idx, via = min(hits, key=lambda h: h[0])
        via = via.copy()
        via[2] = np.clip(via[2], self.Z_LIM[0], self.Z_LIM[1])
        return idx, via

    def _avoid_obstacles(
        self,
        pts: list[NDArray],
        movable: list[bool],
        normals: list[NDArray | None],
        v0: NDArray,
        obs: dict[str, NDArray],
    ) -> tuple[list[NDArray], list[NDArray | None]]:
        """Push waypoints out of obstacles and insert via-points where the trajectory collides."""
        pts = [p.copy() for p in pts]
        movable, normals = list(movable), list(normals)
        inserted = [False] * len(pts)
        poles = np.asarray(obs["obstacles_pos"])[:, :2]
        seen = np.asarray(obs["obstacles_visited"], dtype=bool)
        radii = self.R_SAFE + np.where(seen, 0.0, self.UNC_MARGIN)

        def collision_free(trial_pts: list[NDArray], trial_normals: list[NDArray | None]) -> bool:
            t, poly = self._build_trajectory(trial_pts, trial_normals, v0)
            return (
                self._first_violation(poly(np.linspace(0.0, t[-1], max(80, int(t[-1] * 80)))), obs)
                is None
            )

        for _ in range(60):
            for k, p in enumerate(pts):  # Move waypoints that sit inside a pole's keep-out zone
                if not movable[k]:
                    continue
                for o, r in zip(poles, radii):
                    d = p[:2] - o
                    dist = float(np.linalg.norm(d))
                    if dist < r:
                        u = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
                        p[:2] = o + u * (r + 0.06)
            t, poly = self._build_trajectory(pts, normals, v0)
            ts = np.linspace(0.0, t[-1], max(80, int(t[-1] * 80)))
            hit = self._first_violation(poly(ts), obs)
            if hit is None:
                break
            i, via = hit
            if min(float(np.linalg.norm(via - p)) for p in pts) < 0.06:
                break  # The fix would duplicate an existing waypoint: no further progress possible
            k = int(np.clip(np.searchsorted(t, ts[i], side="right") - 1, 0, len(pts) - 2))
            pts.insert(k + 1, via)
            movable.insert(k + 1, True)
            normals.insert(k + 1, None)
            inserted.insert(k + 1, True)
        else:
            return pts, normals  # Did not converge: keep everything, pruning cannot be verified

        # Prune via-points that turned out to be unnecessary (later insertions often make earlier
        # ones redundant), which removes zig-zags from the path.
        for k in range(len(pts) - 2, 0, -1):
            if inserted[k] and collision_free(
                pts[:k] + pts[k + 1 :], normals[:k] + normals[k + 1 :]
            ):
                del pts[k], normals[k], movable[k], inserted[k]
        return pts, normals

    def _blocked(
        self, P: NDArray, obs: dict[str, NDArray], extra: float = 0.0, plate: bool = False
    ) -> NDArray:
        """Boolean mask of points in P (N, 3) that lie inside a keep-out zone of the track."""
        P = np.atleast_2d(P)
        bad = np.zeros(len(P), dtype=bool)
        seen = np.asarray(obs["obstacles_visited"], dtype=bool)
        for o, sn in zip(np.asarray(obs["obstacles_pos"])[:, :2], seen):
            r = self.R_SAFE + (0.0 if sn else self.UNC_MARGIN) + extra
            bad |= np.linalg.norm(P[:, :2] - o, axis=1) < r
        gate_seen = np.asarray(obs["gates_visited"], dtype=bool)
        for c, q, sn in zip(np.asarray(obs["gates_pos"]), np.asarray(obs["gates_quat"]), gate_seen):
            unc = 0.0 if sn else self.UNC_MARGIN
            dxy = np.linalg.norm(P[:, :2] - c[:2], axis=1)
            bad |= (P[:, 2] < c[2] - 0.25) & (dxy < self.R_STAND + unc + extra)
            local = R.from_quat(q).inv().apply(P - c)
            r_inf = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
            bad |= (
                (np.abs(local[:, 0]) < self.GATE_SLAB + extra)
                & (r_inf > (-1.0 if plate else self.GATE_OPEN))
                & (r_inf < self.GATE_OUTER + unc + extra)
            )
        return bad

    def _route(
        self,
        pts: list[NDArray],
        movable: list[bool],
        normals: list[NDArray | None],
        obs: dict[str, NDArray],
    ) -> tuple[list[NDArray], list[bool], list[NDArray | None]]:
        """Insert A* detour points between distant consecutive waypoints."""
        res = self.GRID_RES
        lo = np.array([-self.XY_LIM[0] - 0.1, -self.XY_LIM[1] - 0.1, 0.05])
        hi = np.array([self.XY_LIM[0] + 0.1, self.XY_LIM[1] + 0.1, self.Z_LIM[1]])
        shape = np.ceil((hi - lo) / res).astype(int) + 1
        axes = [lo[d] + res * np.arange(shape[d]) for d in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        occ = self._blocked(grid.reshape(-1, 3), obs, self.A_INFLATE, plate=True).reshape(shape)
        offsets = [
            o for o in itertools.product((-1, 0, 1), repeat=3) if 0 < sum(abs(x) for x in o) <= 2
        ]
        costs = [res * float(np.linalg.norm(o)) for o in offsets]

        def cell(p: NDArray) -> tuple[int, int, int]:
            return tuple(int(x) for x in np.clip(np.rint((p - lo) / res), 0, shape - 1))

        def free_line(a: NDArray, b: NDArray) -> bool:
            n = max(2, int(np.linalg.norm(b - a) / 0.03))
            s = a + (b - a) * np.linspace(0, 1, n)[:, None]
            return not self._blocked(s, obs, self.A_INFLATE * 0.5, plate=True).any()

        def astar(a: NDArray, b: NDArray) -> list[NDArray] | None:
            ca, cb = cell(a), cell(b)
            gb = np.array(cb)
            open_set = [(0.0, 0.0, ca)]
            g_best, parent = {ca: 0.0}, {ca: None}
            expanded = 0
            while open_set:
                _, g, cur = heapq.heappop(open_set)
                if g > g_best.get(cur, 1e9) + 1e-9:
                    continue
                if cur == cb:
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return [lo + res * np.array(c) for c in path[::-1]]
                expanded += 1
                if expanded > 40000:
                    return None
                for o, cst in zip(offsets, costs):
                    nb = (cur[0] + o[0], cur[1] + o[1], cur[2] + o[2])
                    if not (
                        0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]
                    ):
                        continue
                    near_end = (
                        max(abs(nb[d] - ca[d]) for d in range(3)) <= 1
                        or max(abs(nb[d] - cb[d]) for d in range(3)) <= 1
                    )
                    if occ[nb] and not near_end:
                        continue
                    ng = g + cst
                    if ng < g_best.get(nb, 1e9):
                        g_best[nb] = ng
                        parent[nb] = cur
                        h = res * float(np.linalg.norm(np.array(nb) - gb))
                        heapq.heappush(open_set, (ng + self.A_WEIGHT * h, ng, nb))
            return None

        out_p, out_m, out_n = [pts[0]], [movable[0]], [normals[0]]
        for k in range(1, len(pts)):
            a, b = pts[k - 1], pts[k]
            if np.linalg.norm(b - a) > 0.3 and not free_line(a, b):
                path = astar(a, b)
                if path is not None:
                    path = [a] + path[1:-1] + [b]
                    # Greedy line-of-sight simplification
                    simp, i = [path[0]], 0
                    while i < len(path) - 1:
                        j = len(path) - 1
                        while j > i + 1 and not free_line(path[i], path[j]):
                            j -= 1
                        simp.append(path[j])
                        i = j
                    for q in simp[1:-1]:
                        out_p.append(np.array(q))
                        out_m.append(True)
                        out_n.append(None)
            out_p.append(b)
            out_m.append(movable[k])
            out_n.append(normals[k])
        return out_p, out_m, out_n

    def _plan(self, obs: dict[str, NDArray], start: NDArray, v0: NDArray):
        """(Re-)plan the trajectory from ``start`` through all remaining gates."""
        pts, movable, normals = self._waypoints(obs, start)
        pts, movable, normals = self._route(pts, movable, normals, obs)
        pts, normals = self._avoid_obstacles(pts, movable, normals, v0, obs)
        # Drop non-gate waypoints that sit practically on top of their predecessor.
        clean, clean_normals = [pts[0]], [normals[0]]
        for p, nrm in zip(pts[1:], normals[1:]):
            if np.linalg.norm(p - clean[-1]) > 0.04 or nrm is not None:
                clean.append(p)
                clean_normals.append(nrm)
        self._t_knots, self._spline = self._build_trajectory(
            clean, clean_normals, v0, scale_time=True
        )
        self._pts = np.asarray(clean)
        ts = np.linspace(0.0, self._t_knots[-1], max(80, int(self._t_knots[-1] * 80)))
        self.plan_log.append(
            (self._tick, self._first_violation(self._spline(ts), obs) is not None, len(clean))
        )
        self._t_start = self._tick / self._freq
        self._last_replan_tick = self._tick
        self.n_replans += 1
        self._plan_gates = np.array(obs["gates_pos"], dtype=float)
        self._plan_obstacles = np.array(obs["obstacles_pos"], dtype=float)

    def _needs_replan(self, obs: dict[str, NDArray]) -> bool:
        """Check if the observed track differs from the one the current plan is based on."""
        if self._tick - self._last_replan_tick < 5:
            return False
        gates_moved = np.max(np.abs(np.asarray(obs["gates_pos"]) - self._plan_gates))
        obst_moved = np.max(np.abs(np.asarray(obs["obstacles_pos"]) - self._plan_obstacles))
        return bool(max(gates_moved, obst_moved) > self.REPLAN_POS_TOL)

    # ------------------------------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------------------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].
        """
        t_now = self._tick / self._freq
        t = t_now - self._t_start
        t_end = self._t_knots[-1]

        pos_meas = np.asarray(obs["pos"], dtype=float)
        ref_pos = self._spline(min(t, t_end))
        track_err = float(np.linalg.norm(pos_meas - ref_pos))

        if self._needs_replan(obs) or (
            track_err > self.MAX_TRACK_ERR and self._tick - self._last_replan_tick > 25
        ):
            if track_err > self.MAX_TRACK_ERR:  # Far off: restart from the measured state
                start, v0 = pos_meas, np.asarray(obs["vel"], dtype=float)
            else:  # Continue smoothly from the current reference
                start, v0 = ref_pos, self._spline(min(t, t_end), 1)
            self._plan(obs, start, v0)
            t, t_end = 0.0, self._t_knots[-1]

        if t >= t_end:
            pos, vel, acc = self._spline(t_end), np.zeros(3), np.zeros(3)
        else:
            pos = self._spline(t)
            vel = self._spline(t, 1)
            acc = self._spline(t, 2)
        self.ref_pos = pos.copy()

        # Slow outer-loop integrator: cancels the steady-state offset of the onboard controller
        # (e.g., from the mass mismatch) so the drone actually hits the gate centers.
        if pos_meas[2] > 0.08:
            self._int += self.KI * (pos - pos_meas) / self._freq
            self._int = np.clip(self._int, -self.I_LIM, self.I_LIM)
        pos = pos + self._int

        if t_now > self.TIMEOUT:
            self._finished = True
        return np.concatenate((pos, vel, acc, self._quat, np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the internal clock.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal clock."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Draw the planned trajectory, the waypoints and the current setpoint."""
        t = self._tick / self._freq - self._t_start
        setpoint = self._spline(min(max(t, 0.0), self._t_knots[-1])).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._spline(np.linspace(0, self._t_knots[-1], 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._pts, rgba=(0.0, 0.0, 1.0, 1.0), size=0.015)
