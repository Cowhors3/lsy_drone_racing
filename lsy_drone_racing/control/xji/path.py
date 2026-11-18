"""Smooth geometric racing line through a waypoint route.

The path is created in two stages:

1. A C2 cubic spline is fitted through the waypoints.
2. The spline is relaxed like an elastic band: it is repeatedly smoothed with a spatial Gaussian
   filter, which widens the turns, and after every iteration projected back onto the constraints.
   Gate constraints pin the path onto the gate axis close to the gate centre, obstacle
   constraints push it out of exclusion discs around poles and gate frames.

The result is sampled at a fixed arc length step and provides position, tangent and curvature
vector for the speed profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from numpy.typing import NDArray


def smoothstep5(x: NDArray | float) -> tuple[NDArray, NDArray, NDArray]:
    """Quintic smooth step from 0 to 1 on [0, 1] with its first two derivatives."""
    x = np.clip(x, 0.0, 1.0)
    return (
        x**3 * (10 - 15 * x + 6 * x**2),
        30 * x**2 * (1 - x) ** 2,
        60 * x * (1 - x) * (1 - 2 * x),
    )


@dataclass
class GatePin:
    """Constraint that keeps the path on a gate axis.

    Attributes:
        knot: Index of the gate centre waypoint.
        centre: Point on the axis the path has to pass through.
        normal: Unit direction of the gate axis.
        half_length: Within this arc length around the centre the path lies on the axis. With
            zero length the path is not pinned and only the tunnel through the gate frame is kept
            open.
        gate: Id of the gate, used to disable its own frame discs during the pass.
    """

    knot: int
    centre: NDArray
    normal: NDArray
    half_length: float
    gate: int = -1


@dataclass
class Discs:
    """Circular exclusion zones in the horizontal plane.

    Attributes:
        centres: Disc centres, shape (n, 2).
        radii: Disc radii, shape (n,).
        owner: Gate id every disc belongs to, -1 for obstacle poles. Shape (n,).
    """

    centres: NDArray
    radii: NDArray
    owner: NDArray


class SmoothPath:
    """Arc length parametrised smooth path through waypoints."""

    def __init__(
        self,
        waypoints: NDArray,
        start_tangent: NDArray | None,
        ds: float,
        sigma: float = 0.0,
        iterations: int = 1,
        pins: tuple[GatePin, ...] | list[GatePin] = (),
        discs: Discs | None = None,
        pass_window: float = 0.5,
        tunnel_width: float = 0.22,
        start_fade: float = 0.8,
        pin_fade: float = 0.4,
    ):
        """Fit and relax the path.

        Args:
            waypoints: Waypoints of shape (n, 3). Consecutive points must not coincide.
            start_tangent: Direction of the path at the first waypoint or None if free.
            ds: Arc length step of the sampled path.
            sigma: Total width of the Gaussian smoothing in metres. Zero disables the relaxation.
            iterations: Number of smoothing / projection iterations.
            pins: Gate constraints.
            discs: Exclusion discs.
            pass_window: Arc length around a gate centre in which the path may go through the
                gate.
            tunnel_width: Half width of the tunnel along the gate axis in which the frame discs
                of the gate that is being passed are ignored.
            start_fade: Arc length over which the smoothed path is blended back onto the original
                start position and tangent.
            pin_fade: Arc length over which the final correction onto the gate axes fades out.
        """
        u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))]
        if start_tangent is not None and np.linalg.norm(start_tangent) > 1e-3:
            bc = ((1, start_tangent / np.linalg.norm(start_tangent)), (2, np.zeros(3)))
        else:
            bc = ((2, np.zeros(3)), (2, np.zeros(3)))
        spline = CubicSpline(u, waypoints, bc_type=bc)
        uu = np.linspace(0.0, u[-1], max(int(u[-1] / 0.004), 10))
        p, dp = spline(uu), spline(uu, 1)
        speed = np.maximum(np.linalg.norm(dp, axis=1), 1e-6)
        s = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(uu))]
        s_knots = np.interp(u, uu, s)
        tangent = dp / speed[:, None]
        ddp = spline(uu, 2)
        curv = (ddp - np.sum(ddp * tangent, axis=1)[:, None] * tangent) / speed[:, None] ** 2
        self.clearance_violation = 0.0
        if sigma > 0:
            grid = np.arange(0.0, s[-1], ds)
            points = np.stack([np.interp(grid, s, p[:, i]) for i in range(3)], axis=1)
            arc = [np.abs(grid - s_knots[pin.knot]) for pin in pins]
            # Weight 1 on the straight section through the gate, fading smoothly to 0 outside.
            pin_weights = [
                (1.0 - smoothstep5((a - pin.half_length) / pin_fade)[0], pin)
                for a, pin in zip(arc, pins)
                if pin.half_length > 0
            ]
            tunnels = [(a < pass_window, pin, tunnel_width) for a, pin in zip(arc, pins)]
            relaxed = self._relax(
                points, ds, sigma, iterations, pin_weights, discs, tunnels, tangent[0], tangent[-1]
            )
            if start_tangent is not None:  # Keep position and direction at the start
                w = (1.0 - smoothstep5(grid / start_fade)[0])[:, None]
                t_new = relaxed[1] - relaxed[0]
                t_new = t_new / max(np.linalg.norm(t_new), 1e-9)
                relaxed = relaxed + w * (
                    points[0] - relaxed[0] + grid[:, None] * (tangent[0] - t_new)
                )
            s_new = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(relaxed, axis=0), axis=1))]
            s_knots = np.interp(s_knots, grid, s_new)
            active = self._active_discs(relaxed, discs, tunnels)
            if active is not None:
                dist = np.linalg.norm(relaxed[:, None, :2] - discs.centres[None], axis=-1)
                self._violation = np.where(active, discs.radii[None] - dist, -np.inf)
                self._violation_points, self._violation_s = relaxed, s_new
                self.clearance_violation = float(self._violation.max())
            d1 = np.gradient(relaxed, axis=0)
            tangent = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
            curv = np.gradient(tangent, axis=0) / np.maximum(np.gradient(s_new), 1e-6)[:, None]
            s, p = s_new, relaxed
        # Resample everything on a uniform arc length grid
        self.ds = ds
        self.s = np.arange(0.0, s[-1], ds)
        self.length = float(s[-1])
        self.points = self._resample(self.s, s, p)
        tangent = self._resample(self.s, s, tangent)
        self.tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
        self.curvature_vec = self._resample(self.s, s, curv)
        self.curvature = np.linalg.norm(self.curvature_vec, axis=1)
        self.curvature_rate = np.linalg.norm(np.gradient(self.curvature_vec, ds, axis=0), axis=1)
        self.s_knots = s_knots

    @staticmethod
    def _active_discs(
        points: NDArray, discs: Discs | None, tunnels: list[tuple[NDArray, GatePin, float]]
    ) -> NDArray | None:
        """Mask of shape (n_samples, n_discs) telling where every disc constrains the path.

        The frame discs of a gate are switched off for the samples that fly through this gate,
        i.e. samples close to the gate centre along the path that are inside the gate tunnel.
        """
        if discs is None or len(discs.radii) == 0:
            return None
        active = np.ones((len(points), len(discs.radii)), bool)
        for window, pin, width in tunnels:
            rel = points[:, :2] - pin.centre[:2]
            lateral = np.abs(rel[:, 0] * pin.normal[1] - rel[:, 1] * pin.normal[0])
            active[np.ix_(window & (lateral < width), discs.owner == pin.gate)] = False
        return active

    @staticmethod
    def _relax(
        points: NDArray,
        ds: float,
        sigma: float,
        iterations: int,
        pin_weights: list[tuple[NDArray, GatePin]],
        discs: Discs | None,
        tunnels: list[tuple[NDArray, GatePin, float]],
        tangent0: NDArray,
        tangent1: NDArray,
    ) -> NDArray:
        """Elastic band relaxation: alternate smoothing and projection onto the constraints."""
        n_ext = int(3 * sigma / ds) + 2
        ext = np.arange(1, n_ext + 1)[:, None] * ds
        sigma_it = sigma / np.sqrt(iterations)

        def off_axis(pts: NDArray, pin: GatePin) -> NDArray:
            rel = pts - pin.centre
            return rel - (rel @ pin.normal)[..., None] * pin.normal

        relaxed = points.copy()
        for it in range(iterations):
            if it > 0:
                for weight, pin in pin_weights:  # Straight section through the gate
                    on_axis = weight >= 1.0
                    relaxed[on_axis] -= off_axis(relaxed[on_axis], pin)
                active = SmoothPath._active_discs(relaxed, discs, tunnels)
                if active is not None:
                    for j in range(len(discs.radii)):
                        rel = relaxed[:, :2] - discs.centres[j]
                        dist = np.linalg.norm(rel, axis=1)
                        inside = (dist < discs.radii[j]) & active[:, j]
                        if inside.any():
                            scale = discs.radii[j] / np.maximum(dist[inside], 1e-6)
                            relaxed[inside, :2] = discs.centres[j] + rel[inside] * scale[:, None]
            padded = np.r_[relaxed[0] - ext[::-1] * tangent0, relaxed, relaxed[-1] + ext * tangent1]
            relaxed = gaussian_filter1d(padded, sigma_it / ds, axis=0, mode="nearest")[n_ext:-n_ext]
        # The last smoothing pulls the path slightly off the gate centres again. Shift the path
        # back as a whole around every gate. The shift fades out slowly and adds no kinks.
        for weight, pin in pin_weights:
            centre = int(np.argmax(weight >= 1.0) + np.sum(weight >= 1.0) // 2)
            relaxed = relaxed - weight[:, None] * off_axis(relaxed[centre], pin)
        return relaxed

    def worst_violation(self, margin: float, s_min: float) -> tuple[int, NDArray, float] | None:
        """Find the path sample that intrudes most into an exclusion disc.

        Args:
            margin: Intrusions below this depth are tolerated.
            s_min: Samples closer than this to either end of the path are ignored.

        Returns:
            The disc index, the violating path point and its arc length, or None if the path is
            clear.
        """
        if self.clearance_violation <= margin:
            return None
        s = self._violation_s
        valid = (s > s_min) & (s < s[-1] - 0.05)
        v = np.where(valid[:, None], self._violation, -np.inf)
        i, j = np.unravel_index(np.argmax(v), v.shape)
        if v[i, j] <= margin:
            return None
        return int(j), self._violation_points[i].copy(), float(self._violation_s[i])

    @staticmethod
    def _resample(grid: NDArray, s: NDArray, values: NDArray) -> NDArray:
        return np.stack([np.interp(grid, s, values[:, i]) for i in range(values.shape[1])], axis=1)

    def at(self, s: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Interpolate position, tangent and curvature vector at arc length ``s``.

        Beyond the end the path is continued as a straight line.
        """
        f = np.clip(np.asarray(s) / self.ds, 0, len(self.s) - 1.000001)
        i = f.astype(int)
        w = (f - i)[..., None]
        p = self.points[i] * (1 - w) + self.points[i + 1] * w
        t = self.tangent[i] * (1 - w) + self.tangent[i + 1] * w
        k = self.curvature_vec[i] * (1 - w) + self.curvature_vec[i + 1] * w
        extra = np.maximum(np.asarray(s) - self.s[-1], 0.0)[..., None]
        return p + extra * self.tangent[-1], t, k
