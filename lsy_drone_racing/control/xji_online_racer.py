"""XJI online racer: races tracks whose layout is only known at runtime (level 3).

Pipeline, executed at start and whenever new information about the track arrives:

1. Route: a visibility graph planner finds collision-free waypoints through the next gates.
2. Path: the waypoints are turned into a smooth racing line (spline + elastic band relaxation).
3. Timing: a near time-optimal speed profile respects thrust, lateral acceleration and jerk limits.
4. Tracking: a preview MPC on an identified acceleration response model tracks the reference.

Gates and obstacles report their nominal pose until they are in sensor range. When the true pose
of the next gate is revealed, the reference is shifted onto the true gate centre with a smooth
offset. When a revealed object invalidates the path, or a gate has been passed, the remaining
route is replanned from the current reference state and blended in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.xji.mpc import FIR_XY, FIR_Z, PreviewMPC
from lsy_drone_racing.control.xji.path import Discs, GatePin, SmoothPath, smoothstep5
from lsy_drone_racing.control.xji.route import RouteParams, RoutePlanner
from lsy_drone_racing.control.xji.timing import (
    E_Z,
    GRAVITY,
    BlendedReference,
    OffsetReference,
    Plan,
    QuinticOffset,
    SpeedParams,
    SpeedPlanner,
)
from lsy_drone_racing.control.xji.track import GATE_HALF_WIDTH, Track

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class XJIOnlineRacer(Controller):
    """State-interface racing controller with online route planning and preview MPC."""

    # --- Drone model (identified in simulation) ---
    THRUST_SCALE = 1.611 * 0.0393 / 0.0434  # Onboard thrust rescaling of the feedforward term.
    DRAG = np.array([0.4987, 0.4987, 0.571])  # Linear drag coefficients [1/s].
    THRUST_CLIP = 18.0  # Hard limit on the commanded specific thrust [m/s^2].
    THRUST_VERTICAL_MAX = 15.0  # Limit on the vertical part of the commanded thrust [m/s^2].
    # --- Take-off phase (onboard PD controller with feedforward) ---
    T_START = 0.05  # Time before the reference starts moving [s].
    T_MPC = 0.3  # Time at which the MPC takes over [s].
    TAKEOFF_LEAD = (0.14, 0.06)  # Feedforward lead time horizontal / vertical [s].
    TAKEOFF_ACC_GAIN = np.array([1.08, 1.08, 1.0])
    TAKEOFF_CLIMB = 0.6  # Vertical component of the initial path direction.
    ONBOARD_KP = np.array([0.4, 0.4, 1.25]) * 1.611 / 0.0434
    ONBOARD_KD = np.array([0.2, 0.2, 0.5]) * 1.611 / 0.0434
    # --- Preview MPC ---
    MPC_HORIZON = 30
    MPC_HISTORY = 36
    MPC_XY = dict(qp=1.0, qv=0.15, ru=0.0002, rdu=0.001)
    MPC_Z = dict(qp=1.0, qv=0.02, ru=0.0002, rdu=0.001)
    MPC_GATE_XY = dict(qp=2.0, qv=0.2025, ru=0.0002, rdu=0.00075)  # Tighter gains close to gates.
    MPC_GATE_Z = dict(qp=1.5, qv=0.028, ru=0.0002, rdu=0.001)
    GATE_GAIN_RANGE = (0.3, 1.15)  # Distance to the gate with full / zero weight of the gate gains.
    BIAS_RATE = np.array([0.03, 0.03, 0.02])  # Low-pass rates of the acceleration bias estimate.
    # --- Adaptive reference clock: slow down the reference when the drone falls behind ---
    CLOCK_GAIN = 1.5
    CLOCK_DEADBAND = (0.03, 0.07)  # Tolerated lag / cross-track error [m].
    CLOCK_MIN_RATE = 0.65
    CLOCK_FILTER = 0.2
    # --- Route and path ---
    HORIZON = 3  # Number of upcoming gate passes that are planned.
    DS = 0.01  # Arc length step of the path [m].
    SMOOTHING = 0.35  # Width of the elastic band smoothing [m].
    SMOOTHING_ITERATIONS = 4
    PIN_LENGTH = 0.15  # Half length of the straight section through a gate [m].
    POLE_RADIUS = 0.25  # Path exclusion radius around obstacle poles [m].
    FRAME_RADIUS = 0.2  # Path exclusion radius around gate frame points [m].
    FRAME_OFFSETS = (-0.36, -0.18, 0.0, 0.18, 0.36)  # Lateral positions of the frame points [m].
    FRAME_END_RADIUS = 0.3  # Exclusion radius around the frame ends of the gates in work [m].
    UNSEEN_POLE_MARGIN = 0.1  # Extra clearance to poles that have not been observed yet [m].
    UNSEEN_FRAME_MARGIN = 0.1  # Extra clearance to gates that have not been observed yet [m].
    PASS_WINDOW = 0.5  # Arc length around a gate centre where its own frame is ignored [m].
    TUNNEL_WIDTH = 0.15  # Half width of the free tunnel through a gate [m].
    PATH_REPAIRS = 3  # Maximum number of waypoints inserted to restore clearance.
    PATH_TOLERANCE = 0.02  # Tolerated intrusion of the path into exclusion zones [m].
    START_FADE = 0.8
    # --- Speed profile ---
    APPROACH_SPEED = 1.8  # Speed cap in front of gates whose true pose is unknown [m/s].
    APPROACH_DISTANCE = 0.75  # Length of the capped section in front of the gate [m].
    # --- Online updates ---
    REPLAN_ON_REVEAL = True  # Replan (instead of shifting the reference) when the gate is seen.
    KEEP_APPROACH_SPEED = True  # Do not accelerate towards a gate that was just revealed.
    REVEAL_GAIN = 0.3  # Overshoot of the aim point relative to a late gate position change.
    BLEND_TIME = 0.12  # Duration of the transition to a new plan [s].
    REPLAN_POLE_RADIUS = 0.17  # Replan if the reference gets closer to a revealed pole [m].
    REPLAN_FRAME_RADIUS = 0.17  # Replan if the reference gets closer to a revealed frame [m].
    REPLAN_LOOKAHEAD = 1.5  # Time span of the reference that is checked [s].
    CORRECTION_OVERSHOOT = 1.5  # End velocity factor of the gate correction.
    CORRECTION_MAX_ACC = 14.0  # Acceleration limit of the gate correction [m/s^2].
    CORRECTION_MIN_TIME = 0.06  # Minimum duration of the gate correction [s].
    # --- End of flight: brake to a hover and report finished, so the drone never flies on ---
    GATE_MISS_TIME = 0.3  # Brake if the gate is not passed this long after the reference [s].
    STOP_GRACE = 0.3  # Time past the end of the plan before braking (plan ran out) [s].
    STOP_ACC = 8.0  # Mean deceleration while braking [m/s^2].
    STOP_MIN_TIME = 0.3  # Minimum braking time [s].
    STOP_MARGIN = 0.3  # Distance of the hover point to the safety limits [m].
    STOP_MIN_HEIGHT = 0.3  # Minimum height of the hover point [m].
    HOLD_TIME = 1.0  # Hover time before the controller reports finished [s].
    MAX_FLIGHT_TIME = 20.0  # Hard limit of the flight time [s].

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the initial trajectory and set up the tracking controller.

        Args:
            obs: Initial observation of the environment.
            info: Initial environment information.
            config: Race configuration.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.route_planner = RoutePlanner(self._route_params())
        self.speed_planner = SpeedPlanner(self._speed_params())
        self.track = Track.from_obs(obs)
        self.passed = int(obs["n_gates_passed"])
        self.finished = False  # All gates passed
        self.stopped = False  # Hover after braking is over: the controller reports finished
        self.stop: QuinticOffset | None = None  # Braking reference, replaces the race reference
        self.steps = 0  # Control steps in real time (``tick`` follows the reference clock)
        self.tick = 0.0
        limits = config.env.track.safety_limits
        self.hover_low = np.array(limits.pos_limit_low, float) + self.STOP_MARGIN
        self.hover_high = np.array(limits.pos_limit_high, float) - self.STOP_MARGIN
        self.hover_low[2] = self.STOP_MIN_HEIGHT
        self.clock_rate = 1.0
        self.yaw = float(R.from_quat(obs["quat"]).as_euler("xyz")[2])
        # Reference = base plan (possibly blending from an older one) + gate correction offset
        self.correction: QuinticOffset | None = None
        self.slow_passes: set[int] = set()  # Passes whose gate was revealed during the approach
        self.aim_offsets: dict[int, NDArray] = {}  # Lag compensation of late gate observations
        start = np.asarray(obs["pos"], float)
        self.base = self._make_plan(start, np.zeros(3), np.zeros(3), self.T_START, first=True)
        # Tracking controller state
        dt = 1 / self.freq
        n, m = self.MPC_HORIZON, self.MPC_HISTORY
        self.mpc = (PreviewMPC(FIR_XY, n, dt, n_history=m, **self.MPC_XY),) * 2 + (
            PreviewMPC(FIR_Z, n, dt, n_history=m, **self.MPC_Z),
        )
        self.mpc_gate = (PreviewMPC(FIR_XY, n, dt, n_history=m, **self.MPC_GATE_XY),) * 2 + (
            PreviewMPC(FIR_Z, n, dt, n_history=m, **self.MPC_GATE_Z),
        )
        self.mpc_on = False
        self.u_history = np.zeros((m, 3))
        self.bias = np.zeros(3)
        self.last_vel: NDArray | None = None

    def _route_params(self) -> RouteParams:
        """Route planner settings, consistent with the exclusion zones of the path."""
        return RouteParams(
            pole_clearance=self.POLE_RADIUS + 0.06,
            pole_ring=self.POLE_RADIUS + 0.13,
            wall_clearance=self.FRAME_RADIUS + 0.02,
            wall_ring=self.FRAME_RADIUS + 0.12,
            unseen_pole_margin=self.UNSEEN_POLE_MARGIN,
            unseen_wall_margin=self.UNSEEN_FRAME_MARGIN,
        )

    def _speed_params(self) -> SpeedParams:
        return SpeedParams(drag=tuple(self.DRAG))

    # region Planning

    def _aim_points(self) -> NDArray:
        """Target point inside the gate for every pass of the race."""
        aim = self.track.gates_pos[self.track.sequence].copy()
        for k, offset in self.aim_offsets.items():
            aim[k] += offset
        return aim

    def _exclusion_discs(self) -> Discs:
        """Exclusion zones for the path around poles and gate frames.

        Objects that have not been observed yet may still move and get a larger zone. The ends of
        the frames of the previous and the next gate also get a larger zone: if the drone has to
        turn around such a frame, this enforces a loop it can actually fly at its current speed.
        """
        track = self.track
        poles = track.obstacles_pos[:, :2]
        frames, owner = track.gate_discs(self.FRAME_OFFSETS)
        pole_radii = self.POLE_RADIUS + self.route_planner.pole_margins(track)
        frame_radii = self.FRAME_RADIUS + self.UNSEEN_FRAME_MARGIN * ~track.gates_seen[owner]
        is_end = np.tile(np.abs(self.FRAME_OFFSETS) >= GATE_HALF_WIDTH - 1e-6, len(track.gates_pos))
        near_gates = track.sequence[max(self.passed - 1, 0) : self.passed + 1]
        turn_around = is_end & np.isin(owner, near_gates)
        frame_radii = np.where(turn_around, self.FRAME_END_RADIUS, frame_radii)
        return Discs(
            np.vstack([poles, frames]),
            np.r_[pole_radii, frame_radii],
            np.r_[np.full(len(poles), -1), owner],
        )

    def _make_plan(
        self, pos: NDArray, vel: NDArray, acc: NDArray, t0: float, first: bool = False
    ) -> Plan:
        """Plan route, path and time law from the given state through the next gates.

        Args:
            pos: Start position.
            vel: Start velocity.
            acc: Start acceleration.
            t0: Start time of the plan.
            first: True for the initial plan from the ground.

        Returns:
            The timed plan.
        """
        track, aim = self.track, self._aim_points()
        route = self.route_planner.plan(track, pos, vel, self.passed, self.HORIZON, aim)
        points = route.points
        # Waypoints that open a tunnel through a gate frame: (pass index, waypoint index, pin half
        # length). The gate centres are pinned to the gate axis. Right behind a gate the plan
        # starts inside the tunnel of this gate, which is kept open without a pin.
        knots = [(k, i, self.PIN_LENGTH) for k, i in route.centres.items()]
        if route.exit is not None:
            knots.append((route.exit[0], 0, 0.0))
        tangent = vel if np.linalg.norm(vel) > 0.2 else None
        if first:
            direction = points[1] - points[0]
            tangent = direction / np.linalg.norm(direction) + self.TAKEOFF_CLIMB * E_Z
        discs = self._exclusion_discs()
        best: tuple[SmoothPath, list] | None = None
        for attempt in range(self.PATH_REPAIRS + 1):
            pins = [
                GatePin(i, points[i], track.pass_frame(k)[1], length, int(track.sequence[k]))
                for k, i, length in knots
            ]
            path = SmoothPath(
                points, tangent, self.DS, self.SMOOTHING, self.SMOOTHING_ITERATIONS, pins, discs,
                self.PASS_WINDOW, self.TUNNEL_WIDTH, self.START_FADE,
            )  # fmt: skip
            if best is None or path.clearance_violation < best[0].clearance_violation:
                best = (path, knots)
            violation = path.worst_violation(margin=self.PATH_TOLERANCE, s_min=0.1)
            if violation is None or attempt == self.PATH_REPAIRS:
                break
            # The smoothing pulled the path into an exclusion disc: add a waypoint outside of it.
            disc, point, s_violation = violation
            rel = point[:2] - discs.centres[disc]
            point[:2] = discs.centres[disc] + rel / max(np.linalg.norm(rel), 1e-6) * (
                discs.radii[disc] + 0.02
            )
            new = int(np.clip(np.searchsorted(path.s_knots, s_violation), 1, len(points) - 1))
            points = np.insert(points, new, point, axis=0)
            knots = [(k, i + 1 if i >= new else i, length) for k, i, length in knots]
        path, knots = best
        centres = {k: i for k, i, _ in knots[: len(route.centres)]}
        self.s_centres = {k: float(path.s_knots[i]) for k, i in centres.items()}
        plan = self.speed_planner.plan(path, self._speed_caps(path), vel, acc, t0)
        # The plan is renewed after every gate pass. If its end is reached anyway (e.g. a gate pass
        # was not registered), the drone brakes instead of following the extrapolated path.
        self.plan_end = plan.t0 + plan.duration
        self.t_centres = {k: plan.time_at(s) for k, s in self.s_centres.items()}
        return plan

    def _speed_caps(self, path: SmoothPath) -> NDArray:
        """Limit the speed in front of gates whose true pose has not been observed yet."""
        caps = np.full(len(path.s), self.speed_planner.params.v_max)
        s_prev = 0.0
        for k in sorted(self.s_centres):
            s_gate = self.s_centres[k]
            if not self.track.gates_seen[self.track.sequence[k]] or k in self.slow_passes:
                section = (path.s > max(s_gate - self.APPROACH_DISTANCE, s_prev)) & (
                    path.s <= s_gate + 0.05
                )
                caps[section] = self.APPROACH_SPEED
            s_prev = s_gate
        return caps

    # region Reference

    def _reference(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Evaluate the reference position, velocity and acceleration at the times ``t``."""
        if self.stop is not None:
            return self.stop.eval(t)
        return OffsetReference(self.base, self.correction).eval(t)

    def _replan(self, t: float):
        """Replan from the current reference state and blend from the old to the new plan."""
        old = OffsetReference(self.base, self.correction)
        p, v, a = old.eval(np.array([t]))
        plan = self._make_plan(p[0], v[0], a[0], t)
        self.base = BlendedReference(old, plan, t, self.BLEND_TIME)
        self.correction = None

    def _reference_crossing(self, t: float, k: int) -> tuple[float, NDArray] | None:
        """Find when and where the reference crosses the plane of the k-th gate pass."""
        centre, normal, lateral = self.track.pass_frame(k)
        ts = t + np.arange(0, 2.5, 0.01)
        p, _, _ = self._reference(ts)
        x = (p - centre) @ normal
        for i in np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0)):
            w = -x[i] / (x[i + 1] - x[i])
            crossing = p[i] * (1 - w) + p[i + 1] * w
            if abs((crossing - centre) @ lateral) < 0.6:
                return ts[i] + w * 0.01, crossing
        return None

    def _correct_gate(self, t: float, k: int):
        """Shift the reference so that it crosses the k-th gate at its observed centre."""
        hit = self._reference_crossing(t, k)
        if hit is None:
            return
        t_cross, crossing = hit
        _, normal, _ = self.track.pass_frame(k)
        error = crossing - self._aim_points()[k]
        shift = -(error - (error @ normal) * normal)  # Only correct within the gate plane
        if self.correction is None:
            state = (np.zeros(3), np.zeros(3), np.zeros(3))
        else:
            state = tuple(x[0] for x in self.correction.eval(np.array([t])))
        duration = max(t_cross - t, self.CORRECTION_MIN_TIME)
        for _ in range(2):
            target = state[0] + shift
            end_vel = self.CORRECTION_OVERSHOOT * shift / duration
            offset = QuinticOffset(t, duration, state, target, end_vel)
            a_max = offset.max_acceleration()
            if a_max <= self.CORRECTION_MAX_ACC:
                break
            shift = shift * self.CORRECTION_MAX_ACC / a_max
        self.correction = offset

    # region Track updates

    def _reference_too_close(self, t: float, centres: NDArray, radius: float) -> bool:
        """Check whether the upcoming reference violates the clearance to the given points."""
        ts = t + np.arange(0, self.REPLAN_LOOKAHEAD, 0.02)
        p, _, _ = self._reference(ts)
        dist = np.linalg.norm(p[:, None, :2] - centres[None, :, :2], axis=-1)
        return bool(dist.min() < radius)

    def _update_track(self, obs: dict[str, NDArray], t: float):
        """React to passed gates and to newly revealed gates and obstacles."""
        passed = int(obs["n_gates_passed"])
        gates_seen = np.asarray(obs["gates_visited"], bool)
        obstacles_seen = np.asarray(obs["obstacles_visited"], bool)
        new_gates = gates_seen & ~self.track.gates_seen
        new_obstacles = obstacles_seen & ~self.track.obstacles_seen
        gate_passed = passed != self.passed
        if passed >= self.track.n_passes:
            self.finished = True
            return
        if not (gate_passed or new_gates.any() or new_obstacles.any()):
            return
        nominal = self.track.gates_pos
        self.track = Track.from_obs(obs)
        self.passed = passed
        target = self.track.sequence[passed]
        if new_gates[target]:
            # The drone cannot fully follow a late change of the gate position. Aim beyond the
            # observed centre by a fraction of the change to compensate for the tracking lag.
            _, normal, _ = self.track.pass_frame(passed)
            change = self.track.gates_pos[target] - nominal[target]
            self.aim_offsets[passed] = self.REVEAL_GAIN * (change - (change @ normal) * normal)
        replan = gate_passed
        for j in np.flatnonzero(new_obstacles):
            pole = self.track.obstacles_pos[j : j + 1]
            replan |= self._reference_too_close(t, pole, self.REPLAN_POLE_RADIUS)
        frames, owner = self.track.gate_discs(self.FRAME_OFFSETS)
        for g in np.flatnonzero(new_gates):
            if g != target:
                replan |= self._reference_too_close(t, frames[owner == g], self.REPLAN_FRAME_RADIUS)
        if new_gates[target] and self.KEEP_APPROACH_SPEED:
            self.slow_passes.add(passed)
        if replan or (new_gates[target] and self.REPLAN_ON_REVEAL):
            self._replan(t)
        elif new_gates[target]:
            self._correct_gate(t, passed)

    # region Control

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next state command.

        Args:
            obs: Current observation of the environment.
            info: Optional additional information.

        Returns:
            The state command [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz].
        """
        t = self.tick / self.freq
        pos, vel = np.asarray(obs["pos"], float), np.asarray(obs["vel"], float)
        self._update_clock(t, pos)
        if isinstance(self.base, BlendedReference) and self.base.finished(t):
            self.base = self.base.new
        if self.stop is None:
            self._update_track(obs, t)
            self._check_stop(t)
        acc_now = (vel - self.last_vel) * self.freq if self.last_vel is not None else np.zeros(3)
        if self.mpc_on:
            predicted = np.array(
                [m.predict_acc(self.u_history[:, i]) for i, m in enumerate(self.mpc)]
            )
            self.bias += self.BIAS_RATE * (acc_now - predicted - self.bias)
        elif t >= self.T_MPC:
            self.mpc_on = True
            self.u_history[:] = acc_now / np.array([m.dc_gain for m in self.mpc])
        self.last_vel = vel.copy()
        if self.mpc_on:
            p_cmd, v_cmd, a_cmd, u = self._mpc_command(t, pos, vel)
        else:
            p_cmd, v_cmd, a_cmd, u = self._takeoff_command(t, pos, vel)
        self.u_history = np.roll(self.u_history, 1, axis=0)
        self.u_history[0] = u
        quat = R.from_euler("z", self.yaw).as_quat()
        return np.concatenate((p_cmd, v_cmd, a_cmd, quat, np.zeros(3))).astype(np.float32)

    def _check_stop(self, t: float):
        """Brake to a hover if the target gate was missed or the flight takes too long.

        Normally the environment ends the race at the last gate. If a gate is missed (or its pass
        is not registered), the reference would continue past the last gate along the
        extrapolated path and the drone would keep flying. Instead, it brakes and hovers.
        """
        reason = None
        t_gate = self.t_centres.get(self.passed)
        if t_gate is not None and t > t_gate + self.GATE_MISS_TIME:
            reason = "target gate not passed"
        elif t > self.plan_end + self.STOP_GRACE:
            reason = "end of the plan reached"
        elif self.steps / self.freq > self.MAX_FLIGHT_TIME:
            reason = "maximum flight time exceeded"
        if reason is None:
            return
        logger.warning(f"XJI: braking to a hover ({reason}, {self.passed} gates passed)")
        p, v, a = (x[0] for x in self._reference(np.array([t])))
        duration = max(1.5 * np.linalg.norm(v) / self.STOP_ACC, self.STOP_MIN_TIME)
        target = np.clip(p + 0.5 * duration * v, self.hover_low, self.hover_high)
        self.stop = QuinticOffset(t, duration, (p, v, a), target, np.zeros(3))
        self.correction = None

    def _update_clock(self, t: float, pos: NDArray):
        """Slow the reference clock down while the drone lags behind or is off the path."""
        p, v, _ = self._reference(np.array([t]))
        error, v = pos - p[0], v[0]
        speed = np.linalg.norm(v)
        lag = max(0.0, -error @ v / max(speed, 0.2))
        cross = np.linalg.norm(error - (error @ v) * v / max(speed**2, 0.04))
        excess = max(lag - self.CLOCK_DEADBAND[0], cross - self.CLOCK_DEADBAND[1], 0.0)
        rate = np.clip(1.0 - self.CLOCK_GAIN * excess, self.CLOCK_MIN_RATE, 1.0)
        self.clock_rate += self.CLOCK_FILTER * (rate - self.clock_rate)

    def _gate_gain_weight(self, pos: NDArray) -> float:
        """Blend factor of the tighter MPC gains, 1 close to an observed target gate."""
        if self.passed >= self.track.n_passes:
            return 0.0
        gate = self.track.sequence[self.passed]
        if not self.track.gates_seen[gate]:
            return 0.0
        near, far = self.GATE_GAIN_RANGE
        distance = np.linalg.norm(pos - self.track.gates_pos[gate])
        return float(smoothstep5((far - distance) / (far - near))[0])

    def _mpc_command(
        self, t: float, pos: NDArray, vel: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Track the reference with the preview MPC.

        The state interface is used as an acceleration interface: position and velocity setpoints
        equal the measured state, so the onboard PD terms vanish and only the acceleration
        feedforward acts.

        Returns:
            Position, velocity and acceleration setpoints and the effective acceleration command.
        """
        ts = t + self.clock_rate * np.arange(1, self.MPC_HORIZON + 1) / self.freq
        p_ref, v_ref, a_ref = self._reference(ts)
        v_ref, a_ref = v_ref * self.clock_rate, a_ref * self.clock_rate**2
        weight = self._gate_gain_weight(pos)
        u = np.zeros(3)
        for i in range(3):
            ref = (p_ref[:, i], v_ref[:, i], a_ref[:, i])
            args = (pos[i], vel[i], self.u_history[:, i], self.bias[i], *ref)
            u[i] = self.mpc[i].solve(*args)
            if weight > 0:
                u[i] += weight * (self.mpc_gate[i].solve(*args) - u[i])
        thrust = u + self.DRAG * vel + GRAVITY * E_Z
        # Bound the vertical part so that a height error can never use up the whole thrust and
        # leave the drone without lateral control. Then keep the vertical part, scale the rest.
        thrust[2] = np.clip(thrust[2], 2.0, self.THRUST_VERTICAL_MAX)
        if np.linalg.norm(thrust) > self.THRUST_CLIP:
            horizontal = np.sqrt(self.THRUST_CLIP**2 - thrust[2] ** 2)
            thrust[:2] *= horizontal / max(np.linalg.norm(thrust[:2]), 1e-9)
        u = thrust - self.DRAG * vel - GRAVITY * E_Z
        return pos.copy(), vel.copy(), thrust / self.THRUST_SCALE - GRAVITY * E_Z, u

    def _takeoff_command(
        self, t: float, pos: NDArray, vel: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Follow the reference with the onboard PD controller plus feedforward.

        Used for the first moments of the flight, before the MPC model is valid.

        Returns:
            Position, velocity and acceleration setpoints and the effective acceleration command.
        """
        lead_xy, lead_z = self.TAKEOFF_LEAD
        p_ref, v_ref, a_ref = self._reference(np.array([t, t + lead_xy, t + lead_z]))
        acc = np.r_[a_ref[1, :2], a_ref[2, 2]] + self.DRAG * np.r_[v_ref[1, :2], v_ref[2, 2]]
        thrust = self.TAKEOFF_ACC_GAIN * acc + GRAVITY * E_Z
        thrust[2] = max(thrust[2], 3.0)
        a_cmd = thrust / self.THRUST_SCALE - GRAVITY * E_Z
        u = (
            thrust
            + self.ONBOARD_KP * (p_ref[0] - pos)
            + self.ONBOARD_KD * (v_ref[0] - vel)
            - self.DRAG * vel
            - GRAVITY * E_Z
        )
        return p_ref[0], v_ref[0], a_cmd, u

    # region Callbacks

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the reference clock.

        Returns:
            True once all gates have been passed, or after braking and hovering for
            ``HOLD_TIME``. The deploy script then returns the drone to the start and lands it.
        """
        self.tick += self.clock_rate
        self.steps += 1
        if self.stop is not None:
            t = self.tick / self.freq
            self.stopped = t > self.stop.t0 + self.stop.duration + self.HOLD_TIME
        return self.finished or self.stopped

    def episode_callback(self):
        """Reset the reference clock."""
        self.tick = 0.0
        self.steps = 0
        self.clock_rate = 1.0

    def render_callback(self, sim: Sim):
        """Draw the upcoming reference trajectory."""
        from crazyflow.sim.visualize import draw_line, draw_points

        ts = self.tick / self.freq + np.arange(0, 3.0, 0.05)
        p, _, _ = self._reference(ts)
        draw_line(sim, p, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, p[:1], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
