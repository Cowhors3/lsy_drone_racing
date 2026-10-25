"""Fast racer with re-planning (levels 0-2).

The geometric path comes from the collision-checked planner of :mod:`spline_racer` (waypoints,
A* detours, gate approach/exit points), smoothed by a weighted smoothing spline and
re-parametrized by arc length. A near time-optimal speed profile is computed under a thrust
limit |a + g|, a centripetal and a longitudinal acceleration limit. The reference p(s(t)) is sent
with velocity and acceleration feedforward, corrected for the gains of the onboard controller.

Whenever the exact position of a gate or obstacle is revealed (level 2), path and speed profile
are re-planned from the current reference state. Close to gates whose position is still
unknown the speed is limited so the late correction stays feasible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.spline_racer import SplineRacer

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class FastRacer(Controller):
    """Smoothed planner path + thrust-limited time-optimal speed profile, re-planned on reveal.

    Works with both interfaces (``env.control_mode`` "state" or "attitude"). Level 2, sim,
    120 runs each:
      * attitude, ATT_PROFILE = (2.2, 5.5, 4.5, 2.2): ~73 % success at ~8.6 s (default)
      * attitude, ATT_PROFILE = (2.5, 6.0, 5.0, 2.5): ~67 % at ~8.0 s (risky: ~5 % of
        20-run evaluations end at <= 50 %)
      * state (class defaults): ~73 % at ~9.0 s
      * state, safe: A_LAT = 4.0, A_LON = 3.0, V_UNSEEN = 1.3, R_UNSEEN = 1.0: ~88 % at ~10.1 s
    """

    V_MAX = 2.0  # [m/s] Speed limit along the path.
    A_LAT = 5.0  # [m/s^2] Max. centripetal acceleration (v^2 * curvature).
    A_LON = 4.0  # [m/s^2] Max. acceleration/deceleration along the path.
    V_GATE = 2.0  # [m/s] Speed limit when crossing a gate.
    ATT_PROFILE = (2.2, 5.5, 4.5, 2.2)  # V_MAX, A_LAT, A_LON, V_GATE (attitude interface)
    KNOWN_PROFILE = (2.8, 6.5, 4.5, 2.8, 1.2, 30.0)  # V_MAX, A_LAT, A_LON, V_GATE, F_XY, W_AXIS
    V_UNSEEN = 1.5  # [m/s] Speed limit close to gates whose exact position is not known yet.
    R_UNSEEN = 0.9  # [m] Radius of that zone around the nominal gate position.
    V_TURN = 9.0  # [m/s] Speed limit around gates that are followed by a U-turn.
    R_TURN = 0.5  # [m] Radius of that zone.
    R_POLE = 0.22  # [m] Keep-out radius around obstacle poles (planner R_SAFE).
    A_TOT = 14.0  # [m/s^2] Max. total specific thrust |a + g| (hardware: ~18).
    LAT_SHARE = 0.9  # Share of A_TOT usable for cornering at constant speed.
    T_SMOOTH = 0.0  # [s] Window of the jerk smoothing of the speed profile (0 = off).
    DS = 0.01  # [m] Arc-length resolution of the speed profile.
    LAM = 0.003  # Smoothing of the path (larger = smoother, may cut corners).
    W_CENTER = 1e4  # Smoothing weight of gate centers and of the plan start.
    W_AXIS = 100.0  # Smoothing weight of gate approach/exit points.
    D_DENSE = 0.3  # [m] Max. spacing of smoothing-spline data points along the polyline.
    W_DENSE = 3.0  # Smoothing weight of the densified points.
    CLEAR_SLACK = 0.07  # [m] Allowed intrusion of the smoothed path into the planner margins.
    T_BLEND = 0.1  # [s] Cross-fade time from the old to the new plan after a re-plan.
    D_HEAD = 0.15  # [m] Heading point ahead of the start along the current velocity (re-plans).
    F_XY = 1.3  # Lateral acceleration gain of the onboard controller.
    F_Z = 1.5  # Vertical force gain of the onboard controller.
    ACC_LEAD = 0.1  # [s] Acceleration feedforward lead (attitude lag).
    K_POS = 0.0  # Extra position feedback (multiples of the onboard gain).
    K_VEL = 0.0  # Extra velocity feedback (multiples of the onboard gain).
    FB_POS_LIM = 0.3  # [m] Clip of the position error used for the extra feedback.
    FB_VEL_LIM = 1.0  # [m/s] Clip of the velocity error used for the extra feedback.
    ATT_KP = 25.0  # [1/s^2] Position gain of the own loop (attitude interface).
    ATT_KD = 9.0  # [1/s] Velocity gain of the own loop (attitude interface).
    ATT_KI_Z = 4.0  # [1/s^3] Height integrator of the own loop (attitude interface).
    ATT_LEAD = 0.11  # [s] Acceleration feedforward lead (attitude interface).
    E_LIM = 0.5  # [m] Clip of the position error (attitude interface).
    KI_Z = 1.0  # [1/s] Integral gain on the height error.
    REPLAN_TOL = 0.02  # [m] Observed object displacement that triggers a re-plan.
    MAX_TRACK_ERR = 0.5  # [m] Re-plan from the measured state if the drone is this far off.
    TIMEOUT = 20.0  # [s]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the initial path and speed profile.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._quat = R.from_euler("z", 0.0).as_quat()
        self.ref_pos = np.array(obs["pos"], dtype=float)
        self._int_z = 0.0
        self.n_replans = 0
        self._attitude_mode = config.env.control_mode == "attitude"
        params = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        self._mass = float(params["mass"])
        self._thrust_lim = (4 * float(params["thrust_min"]), 4 * float(params["thrust_max"]))
        self._int_az = 0.0
        self._old: dict | None = None
        self._blend_start = -1.0

        known = not config.env.get("randomizations", {}).get("gate_pos")
        saved = SplineRacer.UNC_MARGIN, SplineRacer.TAKEOFF_Z
        SplineRacer.UNC_MARGIN = 0.0 if known else saved[0]  # Exact positions on level 0/1
        SplineRacer.TAKEOFF_Z = None  # Take off diagonally, no vertical stub
        try:
            self._planner = SplineRacer(obs, info, config)
        finally:
            SplineRacer.UNC_MARGIN, SplineRacer.TAKEOFF_Z = saved
        self._known = known
        if not known and config.env.control_mode == "attitude":  # Stiffer own loop: fly faster
            self.V_MAX, self.A_LAT, self.A_LON, self.V_GATE = self.ATT_PROFILE
        if known:  # Exact track (level 0/1): no reveal surprises, fly the faster profile
            self.V_MAX, self.A_LAT, self.A_LON, self.V_GATE, self.F_XY, self.W_AXIS = (
                self.KNOWN_PROFILE
            )
        self._planner.R_SAFE = self.R_POLE
        if known:
            self._planner.UNC_MARGIN = 0.0
        self._plan(obs, np.array(obs["pos"], dtype=float), np.zeros(3))

    # ------------------------------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------------------------------
    def _plan(self, obs: dict[str, NDArray], start: NDArray, v0: NDArray):
        """(Re-)plan path and speed profile from ``start`` with velocity ``v0``."""
        pl = self._planner
        pts, movable, normals = pl._waypoints(obs, start)
        speed0 = float(np.linalg.norm(v0))
        if speed0 > 0.3:  # Keep the current heading: add a point ahead along the velocity.
            head = start + v0 / speed0 * self.D_HEAD
            if not pl._blocked(head[None], obs, -0.05).any() and (
                len(pts) < 2 or np.linalg.norm(pts[1] - start) > 1.5 * self.D_HEAD
            ):
                pts.insert(1, head)
                movable.insert(1, False)
                normals.insert(1, None)
        pts, _, _ = pl._route(pts, movable, normals, obs)
        wp = np.asarray(pts)
        heads = np.zeros(len(wp), dtype=bool)
        if speed0 > 0.3 and len(wp) > 1:
            heads[1] = np.linalg.norm(wp[1] - (start + v0 / speed0 * self.D_HEAD)) < 1e-6
        keep = np.concatenate(([True], np.linalg.norm(np.diff(wp, axis=0), axis=1) > 0.03))
        wp, heads = wp[keep], heads[keep]
        self.waypoints = wp  # For diagnostics/rendering

        # Smoothing spline: gate centers (and their approach/exit points) are held tightly,
        # detour points only loosely, which removes kinks from the A* polyline. Long segments
        # are densified so the curve cannot bulge away from the (collision-free) polyline.
        gates = np.asarray(obs["gates_pos"])
        dist_gate = np.min(np.linalg.norm(wp[:, None] - gates[None], axis=2), axis=1)
        w = np.where(dist_gate < 0.02, self.W_CENTER, np.where(dist_gate < 0.35, self.W_AXIS, 1.0))
        w[0] = self.W_CENTER
        w[heads] = self.W_CENTER
        dense, dense_w = [wp[0]], [w[0]]
        for k in range(1, len(wp)):
            n_sub = int(np.ceil(np.linalg.norm(wp[k] - wp[k - 1]) / self.D_DENSE))
            for j in range(1, n_sub):
                dense.append(wp[k - 1] + (wp[k] - wp[k - 1]) * j / n_sub)
                dense_w.append(self.W_DENSE)
            dense.append(wp[k])
            dense_w.append(w[k])
        wp, w = np.asarray(dense), np.asarray(dense_w)
        s_knots = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))))
        fine = np.linspace(0.0, s_knots[-1], max(int(s_knots[-1] / 0.05) + 1, 5))
        lam = self.LAM
        for _ in range(4):
            if len(wp) >= 5:
                smooth = np.array(
                    [make_smoothing_spline(s_knots, wp[:, d], w=w, lam=lam)(fine) for d in range(3)]
                ).T
            else:  # Too few points for a smoothing spline: interpolate
                smooth = CubicSpline(s_knots, wp, bc_type="natural")(fine)
            if not pl._blocked(
                smooth[2:], obs, -self.CLEAR_SLACK
            ).any() and not self._reverse_cross(smooth, obs):
                break
            lam /= 4.0  # Collides: follow the polyline more closely
        else:  # Still colliding: interpolate the (collision-free) densified polyline exactly
            smooth = CubicSpline(s_knots, wp, bc_type="natural")(fine)
        s_new = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))))
        keep = np.concatenate(([True], np.diff(s_new) > 1e-6))
        self._path = CubicSpline(s_new[keep], smooth[keep], bc_type="natural")
        self._length = float(s_new[keep][-1])
        self._speed_profile(obs, speed0)
        self._t0 = self._tick / self._freq
        self._cur = self._snapshot()
        self._plan_gates = np.array(obs["gates_pos"], dtype=float)
        self._plan_obstacles = np.array(obs["obstacles_pos"], dtype=float)
        self.n_replans += 1

    def _reverse_cross(self, P: NDArray, obs: dict[str, NDArray]) -> bool:
        """True if the curve P crosses a gate plane (inside the frame) against its direction."""
        for g, (c, q) in enumerate(
            zip(np.asarray(obs["gates_pos"]), np.asarray(obs["gates_quat"]))
        ):
            dirs = self._planner._gate_dir[self._planner._gate_seq == g]
            local = R.from_quat(q).inv().apply(P - c)
            r_inf = np.maximum(np.abs(local[:, 1]), np.abs(local[:, 2]))
            sign = np.sign(local[:, 0])
            cross = np.where((sign[:-1] != sign[1:]) & (r_inf[:-1] < 0.6))[0]
            for i in cross:
                d = np.sign(local[i + 1, 0] - local[i, 0])
                if len(dirs) == 0 or not np.any(dirs == d):
                    return True
        return False

    def _speed_profile(self, obs: dict[str, NDArray], v_start: float):
        """Compute the speed profile v(s) of the current path.

        Total specific force |a + g| stays below A_TOT (thrust limit), the centripetal part below
        A_LAT and the tangential part below A_LON.
        """
        sg = np.arange(0.0, self._length, self.DS)
        if len(sg) < 2:
            sg = np.array([0.0, self._length])
        d1, d2 = self._path(sg, 1), self._path(sg, 2)
        tan = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
        kvec = d2 - np.sum(d2 * tan, axis=1, keepdims=True) * tan  # curvature vector
        k2 = np.maximum(np.sum(kvec**2, axis=1), 1e-12)
        g = np.array([0.0, 0.0, 9.81])
        b = kvec @ g
        a_cap = self.A_TOT * self.LAT_SHARE
        x_max = (-b + np.sqrt(np.maximum(b**2 - k2 * (9.81**2 - a_cap**2), 0.0))) / k2
        v_lim = np.minimum.reduce(
            [np.full(len(sg), self.V_MAX), np.sqrt(self.A_LAT / np.sqrt(k2)), np.sqrt(x_max)]
        )
        path_pts = self._path(sg)
        gates = np.asarray(obs["gates_pos"])
        seen = np.asarray(obs["gates_visited"], dtype=bool)
        for c, sn in zip(gates, seen):
            near = np.linalg.norm(path_pts - c, axis=1) < 0.15
            v_lim[near] = np.minimum(v_lim[near], self.V_GATE)
            if (
                not sn and not self._known
            ):  # Slow down where the gate may still move: the late correction is feasible
                near = np.linalg.norm(path_pts - c, axis=1) < self.R_UNSEEN
                v_lim[near] = np.minimum(v_lim[near], self.V_UNSEEN)

        for c in self._planner._slow_centers:  # Gates followed by a U-turn: cross slowly
            near = np.linalg.norm(path_pts - c, axis=1) < self.R_TURN
            v_lim[near] = np.minimum(v_lim[near], self.V_TURN)

        def lon_limits(i: int, v_i: float) -> tuple[float, float]:
            """Max. acceleration and braking at sample i for speed v_i (thrust circle)."""
            w = kvec[i] * v_i**2 + g
            tw = float(tan[i] @ w)
            disc = tw**2 - float(w @ w) + self.A_TOT**2
            r = np.sqrt(max(disc, 0.0))
            return min(self.A_LON, max(-tw + r, 0.0)), min(self.A_LON, max(tw + r, 0.0))

        v = v_lim.copy()
        v[0] = v_start
        for i in range(len(sg) - 1):  # forward pass (acceleration)
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * lon_limits(i, v[i])[0] * self.DS))
        for i in range(len(sg) - 2, -1, -1):  # backward pass (braking)
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * lon_limits(i + 1, v[i + 1])[1] * self.DS))
        v = np.maximum(v, 0.05)
        dt = np.diff(sg) * 2 / (v[:-1] + v[1:])
        t_grid = np.concatenate(([0.0], np.cumsum(dt)))
        self._t_grid, self._s_grid, self._v_grid = t_grid, sg, v
        self._a_grid = np.gradient(v, t_grid)  # s_ddot
        if self.T_SMOOTH > 0.0 and t_grid[-1] > 2 * self.T_SMOOTH:
            # Jerk smoothing: moving average of the tangential acceleration in time. The
            # attitude cannot change instantly, so acceleration steps are not trackable.
            h = 0.005
            tu = np.arange(0.0, t_grid[-1], h)
            vu = np.interp(tu, t_grid, v)
            au = np.gradient(vu, h)
            n = max(int(self.T_SMOOTH / h), 1)
            kernel = np.ones(n) / n
            a_s = np.convolve(np.pad(au, (n // 2, n - 1 - n // 2), mode="edge"), kernel, "valid")
            v_s = np.maximum(vu[0] + np.concatenate(([0.0], np.cumsum(a_s[:-1]) * h)), 0.02)
            s_s = np.concatenate(([0.0], np.cumsum(0.5 * (v_s[:-1] + v_s[1:])) * h))
            end = np.searchsorted(s_s, sg[-1])
            if end < len(s_s):
                tu, s_s, v_s, a_s = tu[: end + 1], s_s[: end + 1], v_s[: end + 1], a_s[: end + 1]
            self._t_grid, self._s_grid, self._v_grid, self._a_grid = tu, s_s, v_s, a_s
        self.lap_estimate = float(self._t_grid[-1])

    def _snapshot(self) -> dict:
        """The current plan (path, speed profile and start time)."""
        return {
            "path": self._path,
            "t": self._t_grid,
            "s": self._s_grid,
            "v": self._v_grid,
            "a": self._a_grid,
            "t0": self._t0,
        }

    @staticmethod
    def _eval(plan: dict, t: float) -> tuple[NDArray, NDArray, NDArray]:
        """Position, velocity and acceleration of ``plan`` at time t since its start."""
        t_grid = plan["t"]
        t = min(max(t, 0.0), t_grid[-1])
        s = float(np.interp(t, t_grid, plan["s"]))
        sd = float(np.interp(t, t_grid, plan["v"]))
        sdd = float(np.interp(t, t_grid, plan["a"]))
        path = plan["path"]
        d0, d1, d2 = path(s), path(s, 1), path(s, 2)
        n1 = max(float(np.linalg.norm(d1)), 1e-6)
        tan = d1 / n1  # unit tangent (spline is ~arc-length parametrized)
        if t >= t_grid[-1]:
            return d0, np.zeros(3), np.zeros(3)
        return d0, tan * sd, d2 / n1**2 * sd**2 + tan * sdd

    def _ref(self, t_abs: float, lead: float = 0.0) -> tuple[NDArray, NDArray, NDArray]:
        """Reference at absolute time t_abs (+lead).

        Right after a re-plan the previous plan is blended out so that position, velocity and
        acceleration stay continuous.
        """
        cur = self._eval(self._cur, t_abs + lead - self._cur["t0"])
        if self._old is None or t_abs - self._blend_start >= self.T_BLEND:
            return cur
        old = self._eval(self._old, t_abs + lead - self._old["t0"])
        x = (t_abs - self._blend_start) / self.T_BLEND
        alpha = 1.0 - x * x * (3.0 - 2.0 * x)  # smoothstep weight of the old plan
        return tuple(alpha * o + (1.0 - alpha) * c for o, c in zip(old, cur))

    def _needs_replan(self, obs: dict[str, NDArray]) -> bool:
        """True if an observed gate/obstacle differs from the positions used for the plan."""
        moved_g = np.max(np.abs(np.asarray(obs["gates_pos"]) - self._plan_gates))
        moved_o = np.max(np.abs(np.asarray(obs["obstacles_pos"]) - self._plan_obstacles))
        return bool(max(moved_g, moved_o) > self.REPLAN_TOL)

    # ------------------------------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------------------------------
    def _attitude_command(
        self, obs: dict[str, NDArray], pos: NDArray, vel: NDArray, acc: NDArray
    ) -> NDArray:
        """Own position loop for the attitude interface: [roll, pitch, yaw, collective thrust]."""
        self.ref_pos = pos
        e = np.clip(pos - np.asarray(obs["pos"], dtype=float), -self.E_LIM, self.E_LIM)
        ev = vel - np.asarray(obs["vel"], dtype=float)
        if obs["pos"][2] > 0.08:  # Integrator on z: absorbs the unknown mass (level 1/2)
            self._int_az = float(np.clip(self._int_az + self.ATT_KI_Z * e[2] / self._freq, -3, 3))
        a_cmd = acc + self.ATT_KP * e + self.ATT_KD * ev + np.array([0.0, 0.0, 9.81 + self._int_az])
        a_cmd[2] = max(a_cmd[2], 1.0)  # Never command thrust pointing down
        norm = float(np.linalg.norm(a_cmd))
        z_b = a_cmd / norm
        y_b = np.cross(z_b, [1.0, 0.0, 0.0])
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        rpy = R.from_matrix(np.column_stack([x_b, y_b, z_b])).as_euler("xyz")
        thrust = np.clip(self._mass * norm, self._thrust_lim[0], self._thrust_lim[1])
        if self._tick / self._freq > self.TIMEOUT:
            self._finished = True
        return np.array([*rpy, thrust], dtype=np.float32)

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
        t_abs = self._tick / self._freq
        pos_meas = np.asarray(obs["pos"], dtype=float)
        pos, vel, _ = self._ref(t_abs)
        track_err = float(np.linalg.norm(pos_meas - pos))
        if int(obs["n_gates_passed"]) < len(self._planner._gate_seq) and (
            self._needs_replan(obs) or track_err > self.MAX_TRACK_ERR
        ):
            if track_err > self.MAX_TRACK_ERR:  # Far off: restart from the measured state
                self._old = None
                self._plan(obs, pos_meas, np.asarray(obs["vel"], dtype=float))
            else:  # Continue from the current reference and blend the old plan out
                old = self._snapshot()
                self._plan(obs, pos, vel)
                self._old, self._blend_start = old, t_abs
            pos, vel, _ = self._ref(t_abs)

        if self._attitude_mode:
            return self._attitude_command(obs, pos, vel, self._ref(t_abs, self.ATT_LEAD)[2])
        acc = self._ref(t_abs, self.ACC_LEAD)[2]
        # Thrust-model compensation of the onboard controller (identified in hover/step tests):
        # it produces ~F_Z x the commanded vertical force and ~F_XY x the lateral acceleration.
        acc = np.array([acc[0] / self.F_XY, acc[1] / self.F_XY, (acc[2] + 9.81) / self.F_Z - 9.81])
        self.ref_pos = pos
        # Slow integrator on the height error: cancels the remaining steady z offset.
        if pos_meas[2] > 0.08:
            self._int_z = float(
                np.clip(self._int_z + self.KI_Z * (pos[2] - pos_meas[2]) / self._freq, -0.15, 0.15)
            )
        # Outer-loop gain boost: the onboard position loop is soft (kp/m ~ 10 1/s^2), so extra
        # position and velocity feedback is added to the setpoint (tuned on a circle test).
        vel_meas = np.asarray(obs["vel"], dtype=float)
        pos_fb = self.K_POS * np.clip(pos - pos_meas, -self.FB_POS_LIM, self.FB_POS_LIM)
        vel_fb = self.K_VEL * np.clip(vel - vel_meas, -self.FB_VEL_LIM, self.FB_VEL_LIM)
        pos = pos + pos_fb + np.array([0.0, 0.0, self._int_z])
        vel = vel + vel_fb
        if self._tick / self._freq > self.TIMEOUT:
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
        """Draw the path and the current setpoint."""
        draw_points(
            sim, self._ref(self._tick / self._freq)[0].reshape(1, -1), rgba=(1, 0, 0, 1), size=0.02
        )
        draw_line(sim, self._path(np.linspace(0, self._length, 150)), rgba=(0, 1, 0, 1))
