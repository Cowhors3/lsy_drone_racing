"""Time parametrisation of paths and composition of reference trajectories.

The speed along a [SmoothPath][lsy_drone_racing.control.xji.path.SmoothPath] is chosen as large
as the drone limits allow:

* the specific thrust ``|a + g + drag|`` must stay below a limit,
* lateral acceleration and the rate of change of the curvature ("jerk") are bounded, since the
  attitude loop needs time to tilt the drone,
* speed caps can be imposed on sections of the path, e.g. in front of gates whose exact pose is
  still unknown.

The profile is computed with the classic forward (acceleration) / backward (braking) pass over
the arc length samples. All references expose ``eval(t) -> (position, velocity, acceleration)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lsy_drone_racing.control.xji.path import smoothstep5

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.xji.path import SmoothPath

GRAVITY = 9.81
E_Z = np.array([0.0, 0.0, 1.0])


class Reference(Protocol):
    """Anything that can be evaluated as a reference trajectory."""

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return position, velocity and acceleration at the times ``t``."""


@dataclass
class SpeedParams:
    """Limits used for the speed profile (SI units)."""

    v_max: float = 5.0
    thrust_max: float = 14.0  # Maximum specific thrust |a + g + drag|.
    a_acc: float = 10.0  # Maximum tangential acceleration.
    a_dec: float = 10.0  # Maximum tangential deceleration.
    a_lat: float = 13.0  # Maximum lateral acceleration.
    a_vert: float = 4.0  # Maximum vertical acceleration caused by path curvature.
    a_brake: float = 12.0  # Strongest braking assumed if the start speed is too high for the path.
    jerk_max: float = 200.0  # Bound on v^3 * |d curvature / ds|.
    drag: tuple[float, float, float] = (0.4987, 0.4987, 0.571)  # Linear drag [1/s].
    sigma_t: float = 0.1  # Temporal smoothing of the time law [s].
    dt: float = 0.01  # Sampling time of the time law [s].


class Plan:
    """A path together with its time law ``s(t)``."""

    def __init__(
        self, path: SmoothPath, t0: float, dt: float, s: NDArray, sd: NDArray, sdd: NDArray
    ):
        """Store the sampled time law.

        Args:
            path: Geometric path.
            t0: Start time of the plan.
            dt: Sampling time of the time law.
            s: Arc length samples.
            sd: First time derivative of the arc length.
            sdd: Second time derivative of the arc length.
        """
        self.path, self.t0, self.dt = path, t0, dt
        self.s, self.sd, self.sdd = s, sd, sdd

    @property
    def duration(self) -> float:
        """Duration of the plan in seconds."""
        return self.dt * (len(self.s) - 1)

    def time_at(self, s: float) -> float:
        """Return the time at which the plan reaches the arc length ``s``."""
        return self.t0 + float(np.interp(s, self.s, np.arange(len(self.s)) * self.dt))

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return position, velocity and acceleration at the times ``t``."""
        t = np.asarray(t, float)
        f = np.clip((t - self.t0) / self.dt, 0, len(self.s) - 1.000001)
        i = f.astype(int)
        w = f - i
        s = self.s[i] * (1 - w) + self.s[i + 1] * w
        sd = self.sd[i] * (1 - w) + self.sd[i + 1] * w
        sdd = self.sdd[i] * (1 - w) + self.sdd[i + 1] * w
        late = np.maximum(t - (self.t0 + self.duration), 0.0)
        s = s + late * self.sd[-1]
        p, tangent, curv = self.path.at(s)
        return p, tangent * sd[..., None], curv * (sd**2)[..., None] + tangent * sdd[..., None]


class BlendedReference:
    """Smooth transition from an old to a new reference over a short time window."""

    def __init__(self, old: Reference, new: Reference, t_start: float, duration: float):
        """Blend from ``old`` to ``new`` starting at ``t_start``."""
        self.old, self.new, self.t_start, self.duration = old, new, t_start, duration

    def finished(self, t: float) -> bool:
        """Whether the transition is complete at time ``t``."""
        return t > self.t_start + self.duration

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return position, velocity and acceleration at the times ``t``."""
        p, v, a = self.new.eval(t)
        po, vo, ao = self.old.eval(t)
        w, w1, w2 = smoothstep5((np.asarray(t, float) - self.t_start) / self.duration)
        w, w1, w2 = w[..., None], w1[..., None] / self.duration, w2[..., None] / self.duration**2
        dp, dv, da = p - po, v - vo, a - ao
        return po + w * dp, vo + w * dv + w1 * dp, ao + w * da + 2 * w1 * dv + w2 * dp


class QuinticOffset:
    """Position offset that moves smoothly from its current state to a target and stays there."""

    def __init__(
        self, t0: float, duration: float, start: tuple[NDArray, NDArray, NDArray], target: NDArray,
        end_vel: NDArray,
    ):  # fmt: skip
        """Create the offset.

        Args:
            t0: Start time.
            duration: Duration of the transition.
            start: Offset position, velocity and acceleration at ``t0``.
            target: Offset at the end of the transition.
            end_vel: Offset velocity at the end of the transition (overshoot control).
        """
        b0, v0, a0 = (np.asarray(x, float) for x in start)
        self.t0, self.duration = t0, duration
        T = duration
        m = np.array(
            [[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]]
        )
        rhs = np.stack([target - b0 - v0 * T - a0 / 2 * T**2, end_vel - v0 - a0 * T, -a0])
        high = np.linalg.solve(m, rhs)
        self._coef = np.stack([b0, v0, a0 / 2, high[0], high[1], high[2]])
        self._hold = self._poly(np.array(T))[0]

    def _poly(self, tau: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        c, x = self._coef, np.asarray(tau, float)[..., None]
        p = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))))
        v = c[1] + x * (2 * c[2] + x * (3 * c[3] + x * (4 * c[4] + x * 5 * c[5])))
        a = 2 * c[2] + x * (6 * c[3] + x * (12 * c[4] + x * 20 * c[5]))
        return p, v, a

    def max_acceleration(self) -> float:
        """Largest acceleration magnitude during the transition."""
        _, _, a = self._poly(np.linspace(0.0, self.duration, 15))
        return float(np.max(np.linalg.norm(a, axis=1)))

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return offset position, velocity and acceleration at the times ``t``."""
        t = np.asarray(t, float)
        p, v, a = self._poly(np.clip(t - self.t0, 0.0, self.duration))
        before, after = (t < self.t0)[..., None], (t > self.t0 + self.duration)[..., None]
        p = np.where(after, self._hold, np.where(before, 0.0, p))
        return p, np.where(before | after, 0.0, v), np.where(before | after, 0.0, a)


class OffsetReference:
    """Sum of a base reference and an optional offset trajectory."""

    def __init__(self, base: Reference, offset: QuinticOffset | None = None):
        """Combine ``base`` with ``offset``."""
        self.base, self.offset = base, offset

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return position, velocity and acceleration at the times ``t``."""
        p, v, a = self.base.eval(t)
        if self.offset is not None:
            op, ov, oa = self.offset.eval(t)
            p, v, a = p + op, v + ov, a + oa
        return p, v, a


class SpeedPlanner:
    """Computes near time-optimal time laws for smooth paths."""

    def __init__(self, params: SpeedParams | None = None):
        """Create the planner with the given limits."""
        self.params = params or SpeedParams()

    def plan(self, path: SmoothPath, caps: NDArray, v0: NDArray, a0: NDArray, t0: float) -> Plan:
        """Compute the time law of a path.

        After the forward / backward pass the plan is checked against the thrust limit, because
        the temporal smoothing and the curvature changes are not captured exactly by the passes.
        Where the limit is exceeded, the speed cap is lowered and the profile is recomputed.

        Args:
            path: Geometric path.
            caps: Speed cap for every path sample.
            v0: Velocity at the start of the path.
            a0: Acceleration at the start of the path.
            t0: Start time.

        Returns:
            The timed plan.
        """
        prm = self.params
        caps = caps.copy()
        v_start = float(np.dot(v0, path.tangent[0])) if np.linalg.norm(v0) > 0 else 0.0
        drag = np.asarray(prm.drag)
        for _ in range(4):
            v = self.speed_profile(path, caps, max(v_start, 0.0))
            plan = self.time_law(path, v, v_start, a0, t0)
            _, vel, acc = plan.eval(plan.t0 + np.arange(len(plan.s)) * plan.dt)
            thrust = np.linalg.norm(acc + drag * vel + GRAVITY * E_Z, axis=1)
            bad = thrust > prm.thrust_max * 1.03
            if not bad.any():
                break
            for s_bad in plan.s[bad]:
                near = np.abs(path.s - s_bad) < 0.15
                caps[near] = np.minimum(caps[near], 0.93 * v[near])
        return plan

    def speed_profile(self, path: SmoothPath, caps: NDArray, v0: float) -> NDArray:
        """Forward / backward pass under thrust, lateral acceleration and jerk limits."""
        prm = self.params
        kappa, n, ds = path.curvature, len(path.s), path.ds
        thrust2 = prm.thrust_max**2
        # Largest speed for which gravity plus centripetal acceleration fits into the thrust.
        kz = path.curvature_vec[:, 2]
        k2 = np.maximum(kappa**2, 1e-8)
        disc = (GRAVITY * kz) ** 2 - k2 * (GRAVITY**2 - thrust2)
        v2_thrust = (-GRAVITY * kz + np.sqrt(np.maximum(disc, 0.0))) / k2
        v_limit = np.minimum.reduce(
            [
                caps,
                np.sqrt(np.maximum(v2_thrust, 0.0)),
                np.sqrt(prm.a_lat / np.maximum(kappa, 1e-6)),
                np.sqrt(prm.a_vert / np.maximum(np.abs(kz), 1e-6)),
                np.cbrt(prm.jerk_max / np.maximum(path.curvature_rate, 1e-6)),
                np.full(n, prm.v_max),
            ]
        ).tolist()
        curv, tan = path.curvature_vec.tolist(), path.tangent.tolist()
        drag = float(prm.drag[0])
        a_acc, a_dec = prm.a_acc, prm.a_dec

        def tangential_bounds(i: int, speed: float) -> tuple[float, float]:
            """Maximum acceleration and deceleration along the path at sample i."""
            kx, ky, kz_ = curv[i]
            tx, ty, tz = tan[i]
            v2 = speed * speed
            cx = v2 * kx - drag * speed * tx
            cy = v2 * ky - drag * speed * ty
            cz = v2 * kz_ + GRAVITY - drag * speed * tz
            ct = cx * tx + cy * ty + cz * tz
            d = ct * ct - (cx * cx + cy * cy + cz * cz) + thrust2
            if d <= 0:
                return 0.5, 0.5
            r = math.sqrt(d)
            return min(-ct + r, a_acc), min(ct + r, a_dec)

        v = [0.0] * n
        v[0] = v0
        for i in range(n - 1):
            a_max, _ = tangential_bounds(i, v[i])
            v[i + 1] = min(v_limit[i + 1], math.sqrt(v[i] * v[i] + 2 * max(a_max, 0.3) * ds))
        for i in range(n - 2, 0, -1):
            _, d_max = tangential_bounds(i + 1, v[i + 1])
            v[i] = min(v[i], math.sqrt(v[i + 1] * v[i + 1] + 2 * max(d_max, 0.3) * ds))
        # The start speed is given. If it is too high for the path, brake as hard as possible
        # instead of demanding an impossible jump in speed.
        for i in range(n - 1):
            v_brake = math.sqrt(max(v[i] * v[i] - 2 * prm.a_brake * ds, 0.0))
            if v[i + 1] >= v_brake:
                break
            v[i + 1] = v_brake
        return np.maximum(np.array(v), 0.05)

    def time_law(
        self, path: SmoothPath, v: NDArray, v_start: float, a0: NDArray, t0: float
    ) -> Plan:
        """Integrate the speed profile to ``s(t)`` and smooth it in time.

        The smoothing only changes when the drone is where on the path, the geometry is untouched.
        """
        prm = self.params
        dt_seg = 2 * np.diff(path.s) / np.maximum(v[1:] + v[:-1], 0.05)
        t_s = np.r_[0.0, np.cumsum(dt_seg)]
        t_grid = np.arange(int(t_s[-1] / prm.dt) + 1) * prm.dt
        s = np.interp(t_grid, t_s, path.s)
        # Extend with the motion before the start and a constant speed run-out for the filter.
        n_ext = int(3 * prm.sigma_t / prm.dt) + 2
        a_tangential = float(np.dot(a0, path.tangent[0]))
        t_pre = -np.arange(n_ext, 0, -1) * prm.dt
        s_pre = v_start * t_pre + 0.5 * a_tangential * t_pre**2
        s_post = s[-1] + v[-1] * (np.arange(1, n_ext + 1) * prm.dt)
        s_ext = np.r_[s_pre, s, s_post]
        if prm.sigma_t > 0:
            s_ext = gaussian_filter1d(s_ext, prm.sigma_t / prm.dt, mode="nearest")
        sd = np.gradient(s_ext, prm.dt)
        sdd = np.gradient(sd, prm.dt)
        s = s_ext[n_ext:]
        return Plan(path, t0, prm.dt, s - s[0], sd[n_ext:], sdd[n_ext:])
