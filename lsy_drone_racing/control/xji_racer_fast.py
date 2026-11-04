"""XJI Level 2 racer with an earlier launch and distance-blended gate tracking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from typing import Callable

    from crazyflow import Sim
    from numpy.typing import NDArray
GRAV = 9.81
EZ = np.array([0.0, 0.0, 1.0])


def _quintic(
    b0: NDArray, v0: NDArray, a0: NDArray, b1: NDArray, v1: NDArray, a1: NDArray, T: float
) -> NDArray:
    b0, v0, a0, b1, v1, a1 = (np.asarray(x, float) for x in (b0, v0, a0, b1, v1, a1))
    c0, c1, c2 = (b0, v0, a0 / 2)
    T2, T3, T4, T5 = (T * T, T**3, T**4, T**5)
    m = np.array([[T3, T4, T5], [3 * T2, 4 * T3, 5 * T4], [6 * T, 12 * T2, 20 * T3]])
    rhs = np.stack([b1 - c0 - c1 * T - c2 * T2, v1 - c1 - 2 * c2 * T, a1 - 2 * c2])
    c345 = np.linalg.solve(m, rhs.reshape(3, -1)).reshape((3,) + b0.shape)
    return np.stack([c0, c1, c2, c345[0], c345[1], c345[2]])


def _quintic_eval(c: NDArray, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    t = np.asarray(t, float)[..., None] if c.ndim > 1 else np.asarray(t, float)
    p = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))
    v = c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])))
    a = 2 * c[2] + t * (6 * c[3] + t * (12 * c[4] + t * 20 * c[5]))
    return (p, v, a)


def _smooth5(x: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    x = np.clip(x, 0.0, 1.0)
    return (
        x**3 * (10 - 15 * x + 6 * x**2),
        30 * x**2 * (1 - x) ** 2,
        60 * x * (1 - x) * (1 - 2 * x),
    )


H_XY = (
    0.0083,
    0.0649,
    0.1099,
    0.1232,
    0.1069,
    0.0773,
    0.0531,
    0.0417,
    0.0386,
    0.0362,
    0.0312,
    0.0244,
    0.0176,
    0.0125,
    0.0111,
    0.0153,
    0.0233,
    0.0268,
    0.0161,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0187,
    0.0065,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0048,
)
H_Z = (
    0.0,
    0.0,
    0.1445,
    0.2863,
    0.2608,
    0.124,
    0.0091,
    0.0,
    0.0,
    0.0,
    0.0125,
    0.0439,
    0.054,
    0.0205,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


class _PreviewMPC:
    def __init__(
        self, h: tuple, n: int, dt: float, qp: float, qv: float, ru: float, rdu: float, m_hist: int
    ):
        h = np.asarray(h, float)
        M = m_hist
        h = np.r_[h, np.zeros(max(0, M + 1 - len(h)))][: M + 1]
        Hf, Hp = (np.zeros((n, n)), np.zeros((n, M)))
        for j in range(n):
            for i in range(M + 1):
                idx = j - i
                if idx >= 0:
                    Hf[j, idx] += h[i]
                elif -idx - 1 < M:
                    Hp[j, -idx - 1] += h[i]
        jj = np.arange(1, n + 1)[:, None]
        ll = np.arange(n)[None, :]
        Cv = dt * (ll < jj)
        Cp = dt**2 * np.where(ll < jj, jj - ll - 0.5, 0.0)
        Gp, Gv = (Cp @ Hf, Cv @ Hf)
        D = np.eye(n) - np.eye(n, k=-1)
        Hm = qp * Gp.T @ Gp + qv * Gv.T @ Gv + ru * np.eye(n) + rdu * D.T @ D
        row = np.linalg.solve(Hm, np.eye(n))[0]
        self.ap = qp * row @ Gp.T
        self.av = qv * row @ Gv.T
        self.ar = ru * row
        self.ad = rdu * (row @ D.T)[0]
        self.Pp, self.Pv = (Cp @ Hp, Cv @ Hp)
        self.cp1, self.cv1 = (Cp.sum(axis=1), Cv.sum(axis=1))
        self.j_dt = dt * np.arange(1, n + 1)
        self.h = h
        self.n = n

    def solve(
        self,
        p: float,
        v: float,
        up: NDArray,
        b: float,
        rp: NDArray,
        rv: NDArray,
        ra: NDArray,
        ulast: float,
    ) -> float:
        fp = p + self.j_dt * v + self.Pp @ up + self.cp1 * b
        fv = v + self.Pv @ up + self.cv1 * b
        return self.ap @ (rp - fp) + self.av @ (rv - fv) + self.ar @ ra + self.ad * ulast


class _Path:
    def __init__(
        self,
        pts: NDArray,
        t0: NDArray | None,
        ds: float,
        sig_s: float = 0.0,
        pins: list | tuple = (),
        poles: NDArray | tuple = (),
        r_pole: float = 0.2,
        n_it: int = 1,
        d_pin: float = 0.12,
        fade: float = 0.8,
    ):
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.03:
                keep.append(i)
        self.knot_idx = keep
        pts = pts[keep]
        u = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if t0 is not None and np.linalg.norm(t0) > 0.001:
            bc = ((1, t0 / np.linalg.norm(t0)), (2, np.zeros(3)))
        else:
            bc = ((2, np.zeros(3)), (2, np.zeros(3)))
        cs = CubicSpline(u, pts, bc_type=bc)
        uu = np.linspace(0.0, u[-1], max(int(u[-1] / 0.004), 10))
        p, p1, p2 = (cs(uu), cs(uu, 1), cs(uu, 2))
        sp = np.maximum(np.linalg.norm(p1, axis=1), 1e-06)
        s = np.r_[0.0, np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(uu))]
        tan = p1 / sp[:, None]
        kv = (p2 - np.sum(p2 * tan, axis=1)[:, None] * tan) / sp[:, None] ** 2
        s_knots = np.interp(u, uu, s)
        if sig_s > 0:
            sg = np.arange(0.0, s[-1], ds)
            Pg = np.stack([np.interp(sg, s, p[:, i]) for i in range(3)], axis=1)
            n_e = int(3 * sig_s / ds) + 2
            ext = np.arange(1, n_e + 1)[:, None] * ds
            pin_idx = []
            for pin in pins:
                oi, c, n = pin[:3]
                dp = pin[3] if len(pin) > 3 else d_pin
                if oi not in keep:
                    continue
                sk = s_knots[keep.index(oi)]
                m = np.abs(sg - sk) < dp
                pin_idx.append((m, np.asarray(c), np.asarray(n)))
            Ps = Pg.copy()
            sig_it = sig_s / math.sqrt(n_it)
            for it in range(n_it):
                if it > 0:
                    for m, c, n in pin_idx:
                        d = Ps[m] - c
                        Ps[m] = c + (d @ n)[:, None] * n
                    for q in poles:
                        d = Ps[:, :2] - q[:2]
                        r = np.linalg.norm(d, axis=1)
                        m = r < r_pole
                        if m.any():
                            Ps[m, :2] = q[:2] + d[m] / np.maximum(r[m], 1e-06)[:, None] * r_pole
                Pe = np.r_[Ps[0] - ext[::-1] * tan[0], Ps, Ps[-1] + ext * tan[-1]]
                Ps = gaussian_filter1d(Pe, sig_it / ds, axis=0, mode="nearest")[n_e:-n_e]
            if t0 is not None:
                w = (1.0 - _smooth5(sg / fade)[0])[:, None]
                t_s = (Ps[1] - Ps[0]) / max(np.linalg.norm(Ps[1] - Ps[0]), 1e-09)
                Ps = Ps + w * (Pg[0] - Ps[0] + sg[:, None] * (tan[0] - t_s))
            d1 = np.gradient(Ps, ds, axis=0)
            s2 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))]
            s_knots = np.interp(s_knots, sg, s2)
            s, p = (s2, Ps)
            tan = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
            kv = np.gradient(tan, axis=0) / np.maximum(np.gradient(s2), 1e-06)[:, None]
        self.L = s[-1]
        self.ds = ds
        self.s = np.arange(0.0, self.L, ds)
        self.P = np.stack([np.interp(self.s, s, p[:, i]) for i in range(3)], axis=1)
        T = np.stack([np.interp(self.s, s, tan[:, i]) for i in range(3)], axis=1)
        self.T = T / np.linalg.norm(T, axis=1, keepdims=True)
        self.KV = np.stack([np.interp(self.s, s, kv[:, i]) for i in range(3)], axis=1)
        self.kappa = np.linalg.norm(self.KV, axis=1)
        self.dkappa = np.linalg.norm(np.gradient(self.KV, ds, axis=0), axis=1)
        self.s_knots = s_knots

    def at(self, s: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        f = np.clip(np.asarray(s) / self.ds, 0, len(self.s) - 1.000001)
        i = f.astype(int)
        w = (f - i)[..., None]
        P = self.P[i] * (1 - w) + self.P[i + 1] * w
        T = self.T[i] * (1 - w) + self.T[i + 1] * w
        KV = self.KV[i] * (1 - w) + self.KV[i + 1] * w
        extra = np.maximum(np.asarray(s) - self.s[-1], 0.0)[..., None]
        return (P + extra * self.T[-1], T, KV)


class _Plan:
    def __init__(self, path: _Path, t0: float, dt: float, S: NDArray, SD: NDArray, SDD: NDArray):
        self.path, self.t0, self.dt = (path, t0, dt)
        self.S, self.SD, self.SDD = (S, SD, SDD)

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        f = np.clip((np.asarray(t, float) - self.t0) / self.dt, 0, len(self.S) - 1.000001)
        i = f.astype(int)
        w = f - i
        s = self.S[i] * (1 - w) + self.S[i + 1] * w
        sd = self.SD[i] * (1 - w) + self.SD[i + 1] * w
        sdd = self.SDD[i] * (1 - w) + self.SDD[i + 1] * w
        late = np.maximum(np.asarray(t, float) - (self.t0 + self.dt * (len(self.S) - 1)), 0.0)
        s = s + late * self.SD[-1]
        P, T, KV = self.path.at(s)
        return (P, T * sd[..., None], KV * (sd**2)[..., None] + T * sdd[..., None])


class _SumRef:
    def __init__(self, ref: _Plan | _BlendRef, corr_fn: Callable | None = None):
        self.ref, self.corr_fn = (ref, corr_fn)

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        P, V, A = self.ref.eval(t)
        if self.corr_fn is not None:
            cp, cv, ca = self.corr_fn(t)
            P, V, A = (P + cp, V + cv, A + ca)
        return (P, V, A)


class _BlendRef:
    def __init__(self, old: _SumRef | _Plan, new: _Plan, ts: float, T: float):
        self.old, self.new, self.ts, self.T = (old, new, ts, T)

    def eval(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        P, V, A = self.new.eval(t)
        Po, Vo, Ao = self.old.eval(t)
        w, w1, w2 = _smooth5((np.asarray(t, float) - self.ts) / self.T)
        w, w1, w2 = (w[..., None], w1[..., None] / self.T, w2[..., None] / self.T**2)
        dP, dV, dA = (P - Po, V - Vo, A - Ao)
        return (Po + w * dP, Vo + w * dV + w1 * dP, Ao + w * dA + 2 * w1 * dV + w2 * dP)


class XJIRacerAggro(Controller):
    """Track a thrust-limited racing line with preview MPC."""

    K_FF = 1.611 * 0.0393 / 0.0434
    C_DRAG = np.array([0.4987, 0.4987, 0.571])
    TAU_A = 0.14
    TAU_Z = 0.06
    ACC_GAIN = np.array([1.08, 1.08, 1.0])
    MPC = True
    T_MPC = 0.3
    MPC_N = 30
    MPC_M = 36
    MPC_QP, MPC_QV, MPC_RU, MPC_RDU = (1.0, 0.15, 0.0002, 0.001)
    MPC_QP_Z, MPC_QV_Z, MPC_RU_Z, MPC_RDU_Z = (1.0, 0.02, 0.0002, 0.001)
    B_ALPHA = 0.03
    BZ_ALPHA = 0.02
    F_CLIP = 18.0
    KP_ON = np.array([0.4, 0.4, 1.25]) * 1.611 / 0.0434
    KD_ON = np.array([0.2, 0.2, 0.5]) * 1.611 / 0.0434
    M_CTRL = 0.0393
    T_START = 0.05
    DS = 0.01
    DT = 0.01
    V_MAX = 5.0
    F_MAX = 15.65
    A_ACC = 10.0
    A_DEC = 10.0
    A_LAT = 13.0
    SIGMA = 0.1
    SIG_S = 0.35
    EB_IT = 4
    D_PIN = 0.12
    D_PIN_K = (0.2, 0.12, 0.25, 0.25, 0.12)
    F_LEG = (15.15, 16.15, 15.65, 15.65, 15.65)
    J_MAX = 200.0
    T_BLEND = 0.12
    FADE = 0.8
    D_START_MIN = 0.3
    D_APP = 0.48
    V_APP = (2.6, 3.0, 2.8, 2.8, 9.0)
    V_FINAL = 9.0
    R_PATH = 0.2
    R_REPLAN = 0.17
    Z_OFF = 0.0
    AIM_L = (-0.006, -0.011, -0.056, 0.032, 0.036)
    AIM_Z = (0.0, 0.0, 0.0, 0.0, 0.0)
    CORR_DELAY = 0.0
    CORR_LAMBDA = 1.5
    CORR_GAIN = 1.0
    CORR_AMAX = 14.0
    CORR_TMIN = 0.06
    CORR_EARLY = 0.0
    YAW_ALIGN = False
    T_YAW = 0.35
    ROUTE = (
        (
            ("o", 0, 0, (0.0, -0.24, 0.53)),
            ("g", 0, (-0.15, 0.0, 0.0)),
            ("c", 0),
            ("g", 0, (0.27, 0.0, 0.0)),
        ),
        (
            ("g", 0, (0.65, 0.32, 0.2)),
            ("g", 1, (-0.3, -0.05, -0.05)),
            ("c", 1),
            ("g", 1, (0.15, 0.05, 0.0)),
        ),
        (("g", 2, (-0.35, 0.0, 0.0)), ("c", 2), ("g", 2, (0.12, 0.0, 0.0))),
        (
            ("g", 2, (0.3, 0.48, 0.2)),
            ("o", 3, 3, (0.0, -0.22, 1.15)),
            ("c", 3),
            ("g", 3, (0.25, 0.0, 0.0)),
        ),
        (
            ("g", 1, (-0.6, 0.4, 0.0)),
            ("g", 1, (-0.2, 0.0, 0.05)),
            ("c", 1),
            ("g", 1, (1.0, -0.05, -0.1)),
        ),
    )

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller."""
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self.tick = 0
        self.finished = False
        self.mpc_on = False
        self.seq = np.asarray(obs["gate_sequence"], int)
        self.n_pass = len(self.seq)
        self.passed = int(obs["n_gates_passed"])
        self._read_track(obs)
        self.g_seen = np.asarray(obs["gates_visited"], bool).copy()
        self.o_seen = np.asarray(obs["obstacles_visited"], bool).copy()
        self.corr = None
        self.yaw0 = float(R.from_quat(obs["quat"]).as_euler("xyz")[2])
        self.yaw_from, self.yaw_to, self.t_yaw = (self.yaw0, self.yaw0, 0.0)
        p0 = np.asarray(obs["pos"], float)
        self.p_start = p0.copy()
        self.ri = 0
        self.base = self._make_plan(p0, np.zeros(3), np.zeros(3), self.T_START, first=True)
        self._update_yaw_target(0.0)
        self.u_hist = np.zeros((self.MPC_M, 3))
        self.bias = np.zeros(3)
        self.v_prev = None
        self.mpc_xy = _PreviewMPC(
            H_XY,
            self.MPC_N,
            1 / self.freq,
            self.MPC_QP,
            self.MPC_QV,
            self.MPC_RU,
            self.MPC_RDU,
            self.MPC_M,
        )
        self.mpc_z = _PreviewMPC(
            H_Z,
            self.MPC_N,
            1 / self.freq,
            self.MPC_QP_Z,
            self.MPC_QV_Z,
            self.MPC_RU_Z,
            self.MPC_RDU_Z,
            self.MPC_M,
        )

        self.mpc_gate = _PreviewMPC(
            H_XY,
            self.MPC_N,
            1 / self.freq,
            2.0 * self.MPC_QP,
            1.35 * self.MPC_QV,
            self.MPC_RU,
            0.75 * self.MPC_RDU,
            self.MPC_M,
        )
        self.mpc_gate_z = _PreviewMPC(
            H_Z,
            self.MPC_N,
            1 / self.freq,
            1.5 * self.MPC_QP_Z,
            1.4 * self.MPC_QV_Z,
            self.MPC_RU_Z,
            self.MPC_RDU_Z,
            self.MPC_M,
        )

    def _read_track(self, obs: dict):
        self.gpos = np.asarray(obs["gates_pos"], float).copy()
        self.gyaw = R.from_quat(obs["gates_quat"]).as_euler("xyz")[:, 2]
        self.opos = np.asarray(obs["obstacles_pos"], float).copy()

    def _gframe(self, g: int) -> tuple[NDArray, NDArray, NDArray]:
        c, s = (math.cos(self.gyaw[g]), math.sin(self.gyaw[g]))
        return (self.gpos[g], np.array([c, s, 0.0]), np.array([-s, c, 0.0]))

    def _route_flat(self) -> list:
        return [(k, item) for k in range(self.n_pass) for item in self.ROUTE[k]]

    def _aim(self, k: int) -> NDArray:
        gp, n, lat = self._gframe(self.seq[k])
        return gp + (self.Z_OFF + self.AIM_Z[k]) * EZ + self.AIM_L[k] * lat

    def _item_pos(self, item: tuple, k: int | None = None) -> NDArray:
        if item[0] == "c":
            if k is not None:
                return self._aim(k)
            gp, _, _ = self._gframe(item[1])
            return gp + self.Z_OFF * EZ
        if item[0] == "g":
            gp, n, lat = self._gframe(item[1])
            a, b, dz = item[2]
            return gp + a * n + b * lat + dz * EZ
        _, n, lat = self._gframe(item[2])
        a, b, z = item[3]
        q = self.opos[item[1]] + a * n + b * lat
        return np.array([q[0], q[1], z])

    def _advance_route(self, pos: NDArray):
        flat = self._route_flat()
        while self.ri < len(flat):
            k, item = flat[self.ri]
            if item[0] == "c":
                if self.passed > k:
                    self.ri += 1
                    continue
                break
            if k > self.passed:
                break
            q = self._item_pos(item, k)
            q_prev = self.p_start
            if self.ri > 0:
                q_prev = self._item_pos(flat[self.ri - 1][1], flat[self.ri - 1][0])
            q_next = q
            if self.ri + 1 < len(flat):
                q_next = self._item_pos(flat[self.ri + 1][1], flat[self.ri + 1][0])
            d = q_next - q_prev
            d = d / max(np.linalg.norm(d), 1e-06)
            if np.dot(pos - q, d) > 0.0 or np.linalg.norm(pos - q) < 0.15:
                self.ri += 1
                continue
            break

    def _route_points(self) -> tuple[list, dict]:
        pts, centers = ([], {})
        for k, item in self._route_flat()[self.ri :]:
            if item[0] == "c":
                centers[k] = len(pts)
            pts.append(self._item_pos(item, k))
        return (pts, centers)

    def _make_plan(
        self, p0: NDArray, v0: NDArray, a0: NDArray, t0: float, first: bool = False
    ) -> _Plan:
        pts, centers = self._route_points()
        if not first:
            cidx = set(centers.values())
            keep = [
                i
                for i in range(len(pts))
                if i in cidx or np.linalg.norm(pts[i] - p0) > self.D_START_MIN
            ]
            remap = {old: new for new, old in enumerate(keep)}
            pts = [pts[i] for i in keep]
            centers = {k: remap[i] for k, i in centers.items() if i in remap}
        pts = np.array([p0] + pts)
        centers = {k: i + 1 for k, i in centers.items()}
        t_dir = v0 if np.linalg.norm(v0) > 0.2 else None
        if first:
            d = pts[1] - pts[0]
            t_dir = d / np.linalg.norm(d) + np.array([0.0, 0.0, 0.6])
        path = None
        for _ in range(4):
            path = _Path(
                pts,
                t_dir,
                self.DS,
                self.SIG_S,
                pins=self._pins(pts, centers),
                poles=self.opos,
                r_pole=self.R_PATH,
                n_it=self.EB_IT,
                d_pin=self.D_PIN,
                fade=self.FADE,
            )
            moved = False
            for j in range(len(self.opos)):
                d = path.P[:, :2] - self.opos[j, :2]
                dist = np.linalg.norm(d, axis=1)
                i = int(np.argmin(dist))
                if dist[i] < self.R_PATH - 0.01 and 0.1 < path.s[i] < path.L - 0.05:
                    q = path.P[i].copy()
                    q[:2] = self.opos[j, :2] + d[i] / max(dist[i], 1e-06) * (self.R_PATH + 0.02)
                    kn = np.searchsorted(path.s_knots, path.s[i])
                    orig = path.knot_idx[min(kn, len(path.knot_idx) - 1)]
                    pts = np.insert(pts, orig, q, axis=0)
                    centers = {k: c + 1 if c >= orig else c for k, c in centers.items()}
                    moved = True
                    break
            if not moved:
                break
        s_c = {}
        for k, i in centers.items():
            if i in path.knot_idx:
                s_c[k] = path.s_knots[path.knot_idx.index(i)]
        self.s_centers = s_c
        caps = np.full(len(path.s), self.V_MAX)
        prev_s = 0.0
        for k in sorted(s_c):
            g = self.seq[k]
            cap = self.V_FINAL if k == self.n_pass - 1 and self.g_seen[g] else None
            if not self.g_seen[g]:
                cap = self.V_APP[k]
            if cap is not None:
                m = (path.s > s_c[k] - self.D_APP) & (path.s <= s_c[k] + 0.05) & (path.s >= prev_s)
                caps[m] = np.minimum(caps[m], cap)
            prev_s = s_c[k]
        v_start = float(np.dot(v0, path.T[0])) if np.linalg.norm(v0) > 0 else 0.0
        for _ in range(4):
            v = self._speed_profile(path, caps, max(v_start, 0.0))
            plan = self._time_law(path, v, v_start, a0, t0)
            _, Vr, Ar = plan.eval(plan.t0 + np.arange(len(plan.S)) * plan.dt)
            f = np.linalg.norm(Ar + self.C_DRAG * Vr + GRAV * EZ, axis=1)
            fl = np.sqrt(self._f_leg_arr(path, self.F_MAX))
            s_q = np.clip(plan.S, 0, path.s[-1])
            bad = f > np.interp(s_q, path.s, fl) * 1.03
            if not bad.any():
                break
            sb = plan.S[bad]
            for sv in sb:
                m = np.abs(path.s - sv) < 0.15
                caps[m] = np.minimum(caps[m], 0.93 * v[m])
        return plan

    def _time_law(self, path: _Path, v: NDArray, v_start: float, a0: NDArray, t0: float) -> _Plan:
        ds = np.diff(path.s)
        dt_seg = 2 * ds / np.maximum(v[1:] + v[:-1], 0.05)
        t_s = np.r_[0.0, np.cumsum(dt_seg)]
        t_end = t_s[-1]
        n_t = int(t_end / self.DT) + 1
        tg = np.arange(n_t) * self.DT
        S = np.interp(tg, t_s, path.s)
        n_sig = int(3 * self.SIGMA / self.DT) + 2
        a_t = float(np.dot(a0, path.T[0]))
        th = -np.arange(n_sig, 0, -1) * self.DT
        S_pre = v_start * th + 0.5 * a_t * th**2
        S_post = S[-1] + v[-1] * (np.arange(1, n_sig + 1) * self.DT)
        S_ext = np.r_[S_pre, S, S_post]
        if self.SIGMA > 0:
            S_ext = gaussian_filter1d(S_ext, self.SIGMA / self.DT, mode="nearest")
        S = S_ext[n_sig:]
        SD = np.gradient(S_ext, self.DT)[n_sig:]
        SDD = np.gradient(np.gradient(S_ext, self.DT), self.DT)[n_sig:]
        S = S - S[0]
        return _Plan(path, t0, self.DT, S, SD, SDD)

    def _f_leg_arr(self, path: _Path, f_max: float) -> NDArray:
        F2 = np.full(len(path.s), f_max**2)
        if self.F_LEG is None:
            return F2
        sc = f_max / self.F_MAX
        prev = 0.0
        for k in sorted(self.s_centers):
            m = (path.s >= prev) & (path.s <= self.s_centers[k] + 0.1)
            F2[m] = (self.F_LEG[k] * sc) ** 2
            prev = self.s_centers[k] + 0.1
        last = max(self.s_centers) if self.s_centers else None
        if last is not None:
            F2[path.s > self.s_centers[last] + 0.1] = (self.F_LEG[last] * sc) ** 2
        return F2

    def _speed_profile(self, path: _Path, caps: NDArray, v0: float) -> NDArray:
        kap = path.kappa
        n = len(kap)
        ds = self.DS
        F2a = self._f_leg_arr(path, self.F_MAX)
        kz = path.KV[:, 2]
        k2 = np.maximum(kap**2, 1e-08)
        disc = (GRAV * kz) ** 2 - k2 * (GRAV**2 - F2a)
        v2_f = (-GRAV * kz + np.sqrt(np.maximum(disc, 0.0))) / k2
        vlim = np.minimum.reduce(
            [
                caps,
                np.sqrt(np.maximum(v2_f, 0.0)),
                np.sqrt(self.A_LAT / np.maximum(kap, 1e-06)),
                np.cbrt(self.J_MAX / np.maximum(path.dkappa, 1e-06)),
                np.full(n, self.V_MAX),
            ]
        )
        vlim = vlim.tolist()
        KV = path.KV.tolist()
        T = path.T.tolist()
        cd = float(self.C_DRAG[0])
        aacc, adec = (self.A_ACC, self.A_DEC)
        F2l = F2a.tolist()

        def a_bounds(i: int, vv: float) -> tuple[float, float]:
            kx, ky, kz_ = KV[i]
            tx, ty, tz = T[i]
            v2 = vv * vv
            cx = v2 * kx - cd * vv * tx
            cy = v2 * ky - cd * vv * ty
            cz = v2 * kz_ + GRAV - cd * vv * tz
            ct = cx * tx + cy * ty + cz * tz
            c2 = cx * cx + cy * cy + cz * cz
            d = ct * ct - c2 + F2l[i]
            if d <= 0:
                return (0.5, 0.5)
            r = math.sqrt(d)
            return (min(-ct + r, aacc), min(ct + r, adec))

        v = [0.0] * n
        v[0] = v0
        for i in range(n - 1):
            amax, _ = a_bounds(i, v[i])
            v[i + 1] = min(vlim[i + 1], math.sqrt(v[i] * v[i] + 2 * max(amax, 0.3) * ds))
        for i in range(n - 2, 0, -1):
            _, dmax = a_bounds(i + 1, v[i + 1])
            v[i] = min(v[i], math.sqrt(v[i + 1] * v[i + 1] + 2 * max(dmax, 0.3) * ds))
        return np.maximum(np.array(v), 0.05)

    def _pins(self, pts: NDArray, centers: dict) -> list:
        out = []
        for k, i in centers.items():
            g = self.seq[k]
            _, n, _ = self._gframe(g)
            dp = self.D_PIN_K[k] if self.D_PIN_K is not None else self.D_PIN
            out.append((i, self._aim(k), n, dp))
        return out

    def _corr_eval(self, corr: tuple | None, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        t = np.asarray(t, float)
        z = np.zeros(t.shape + (3,))
        if corr is None:
            return (z, z.copy(), z.copy())
        t0, t1, c, hold = corr
        tau = np.clip(t - t0, 0.0, t1 - t0)
        p, v, a = _quintic_eval(c, tau)
        before = (t < t0)[..., None]
        after = (t > t1)[..., None]
        p = np.where(before, 0.0, p)
        v = np.where(before | after, 0.0, v)
        a = np.where(before | after, 0.0, a)
        p = np.where(after, hold, p)
        return (p, v, a)

    def _ref(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        P, V, A = self.base.eval(t)
        if self.corr is not None:
            cp, cv, ca = self._corr_eval(self.corr, t)
            P, V, A = (P + cp, V + cv, A + ca)
        return (P, V, A)

    def _replan(self, t: float):
        P, V, A = self._ref(np.array([t]))
        c = self.corr
        old = _SumRef(self.base, (lambda tt: self._corr_eval(c, tt)) if c is not None else None)
        plan = self._make_plan(P[0], V[0], A[0], t)
        self.base = _BlendRef(old, plan, t, self.T_BLEND)
        self.corr = None

    def _collapse(self, t: float):
        if isinstance(self.base, _BlendRef) and t > self.base.ts + self.base.T:
            self.base = self.base.new

    def _crossing(self, t: float, k: int) -> tuple[float | None, NDArray | None]:
        g = self.seq[k]
        gp, n, lat = self._gframe(g)
        ts = t + np.arange(0, 2.5, self.DT)
        P, V, _ = self._ref(ts)
        x = (P - gp) @ n
        idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
        for i in idx:
            w = -x[i] / (x[i + 1] - x[i])
            c = P[i] * (1 - w) + P[i + 1] * w
            if abs((c - gp) @ lat) < 0.6:
                return (ts[i] + w * self.DT, c)
        return (None, None)

    def _correct_gate(self, t: float, k: int, obs: dict):
        tg, c = self._crossing(t, k)
        if tg is None:
            return
        g = self.seq[k]
        gp, n, lat = self._gframe(g)
        e = c - self._aim(k)
        d = -(e - e @ n * n)
        cp, cv, ca = self._corr_eval(self.corr, np.array([t]))
        cp, cv, ca = (cp[0], cv[0], ca[0])
        t0 = t + self.CORR_DELAY if np.linalg.norm(cv) + np.linalg.norm(ca) < 1e-06 else t
        T = max(tg - t0 - self.CORR_EARLY, self.CORR_TMIN)
        D = cp + self.CORR_GAIN * d
        v_end = self.CORR_LAMBDA * (D - cp) / T
        coef = _quintic(cp, cv, ca, D, v_end, np.zeros(3), T)
        _, _, aa = _quintic_eval(coef, np.linspace(0, T, 15))
        amax = np.max(np.linalg.norm(aa, axis=1))
        if amax > self.CORR_AMAX:
            f = self.CORR_AMAX / amax
            D = cp + f * (D - cp)
            v_end = self.CORR_LAMBDA * (D - cp) / T
            coef = _quintic(cp, cv, ca, D, v_end, np.zeros(3), T)
        hold = _quintic_eval(coef, np.array([T]))[0][0]
        self.corr = (t0, t0 + T, coef, hold)

    def _obstacle_too_close(self, t: float, j: int) -> bool:
        ts = t + np.arange(0, 1.5, 0.02)
        P, _, _ = self._ref(ts)
        d = np.linalg.norm(P[:, :2] - self.opos[j, :2], axis=1)
        return bool(d.min() < self.R_REPLAN)

    def _update_yaw_target(self, t: float):
        if not self.YAW_ALIGN or self.passed >= self.n_pass:
            return
        g = self.seq[self.passed]
        target = self.gyaw[g]
        cur = self._yaw_at(t)
        k = round((cur - target) / (math.pi / 2))
        target = target + k * math.pi / 2
        if abs(target - self.yaw_to) > 0.001:
            self.yaw_from, self.yaw_to, self.t_yaw = (cur, target, t)

    def _yaw_at(self, t: float) -> float:
        w, _, _ = _smooth5(np.array((t - self.t_yaw) / self.T_YAW))
        return float(self.yaw_from + w * (self.yaw_to - self.yaw_from))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state."""
        t = self.tick / self.freq
        passed = int(obs["n_gates_passed"])
        g_seen = np.asarray(obs["gates_visited"], bool)
        o_seen = np.asarray(obs["obstacles_visited"], bool)
        new_g = g_seen & ~self.g_seen
        new_o = o_seen & ~self.o_seen
        self._collapse(t)
        if passed >= self.n_pass:
            self.finished = True
        if new_g.any() or new_o.any() or passed != self.passed:
            self._read_track(obs)
            self.g_seen, self.o_seen = (g_seen.copy(), o_seen.copy())
            need = passed != self.passed
            for j in np.where(new_o)[0]:
                need |= self._obstacle_too_close(t, j)
            self.passed = passed
            self._advance_route(np.asarray(obs["pos"], float))
            if passed < self.n_pass:
                if need:
                    self._replan(t)
                k = passed
                if self.g_seen[self.seq[k]] and (need or new_g[self.seq[k]]):
                    self._correct_gate(t, k, obs)
            self._update_yaw_target(t)
        pos, vel = (np.asarray(obs["pos"], float), np.asarray(obs["vel"], float))
        a_now = (vel - self.v_prev) * self.freq if self.v_prev is not None else np.zeros(3)
        if self.v_prev is not None and self.mpc_on:
            h_xy, h_z = (self.mpc_xy.h[: self.MPC_M], self.mpc_z.h[: self.MPC_M])
            u_h = self.u_hist
            a_pred = np.r_[h_xy @ u_h[:, 0], h_xy @ u_h[:, 1], h_z @ u_h[:, 2]]
            r = a_now - a_pred
            self.bias[:2] += self.B_ALPHA * (r[:2] - self.bias[:2])
            self.bias[2] += self.BZ_ALPHA * (r[2] - self.bias[2])
        self.v_prev = vel.copy()
        if self.MPC and t >= self.T_MPC and (not self.mpc_on):
            self.mpc_on = True
            sxy, sz = (self.mpc_xy.h[: self.MPC_M].sum(), self.mpc_z.h[: self.MPC_M].sum())
            self.u_hist[:] = a_now / np.array([sxy, sxy, sz])
        if self.mpc_on:
            ts = t + np.arange(1, self.MPC_N + 1) / self.freq
            P, V, A = self._ref(np.r_[t, ts])
            u = np.zeros(3)
            gate_weight = 0.0
            if self.passed < self.n_pass and self.g_seen[self.seq[self.passed]]:
                center = self.gpos[self.seq[self.passed]]
                distance = np.linalg.norm(pos - center)
                gate_weight = float(_smooth5((1.15 - distance) / 0.85)[0])
            for ax in range(3):
                mpc = self.mpc_z if ax == 2 else self.mpc_xy
                u[ax] = mpc.solve(
                    pos[ax],
                    vel[ax],
                    self.u_hist[:, ax],
                    self.bias[ax],
                    P[1:, ax],
                    V[1:, ax],
                    A[1:, ax],
                    self.u_hist[0, ax],
                )
                if gate_weight > 0:
                    gate_mpc = self.mpc_gate_z if ax == 2 else self.mpc_gate
                    gate_u = gate_mpc.solve(
                        pos[ax],
                        vel[ax],
                        self.u_hist[:, ax],
                        self.bias[ax],
                        P[1:, ax],
                        V[1:, ax],
                        A[1:, ax],
                        self.u_hist[0, ax],
                    )
                    u[ax] += gate_weight * (gate_u - u[ax])
            a_thr = u + self.C_DRAG * vel + GRAV * EZ
            a_thr[2] = max(a_thr[2], 2.0)
            if np.linalg.norm(a_thr) > self.F_CLIP:
                hz = a_thr[2]
                if hz >= self.F_CLIP:
                    a_thr = np.array([0.0, 0.0, self.F_CLIP])
                else:
                    sc = math.sqrt(self.F_CLIP**2 - hz**2) / max(np.linalg.norm(a_thr[:2]), 1e-09)
                    a_thr[:2] *= sc
            u_applied = a_thr - self.C_DRAG * vel - GRAV * EZ
            p, v = (pos.copy(), vel.copy())
            a_ff = a_thr / self.K_FF - GRAV * EZ
        else:
            P, V, A = self._ref(np.array([t, t + self.TAU_A, t + self.TAU_Z]))
            p, v = (P[0], V[0])
            a = np.r_[A[1, :2] + self.C_DRAG[:2] * V[1, :2], A[2, 2] + self.C_DRAG[2] * V[2, 2]]
            a_thr = self.ACC_GAIN * a + GRAV * EZ
            a_thr[2] = max(a_thr[2], 3.0)
            a_ff = a_thr / self.K_FF - GRAV * EZ
            u_applied = (
                self.K_FF * (a_ff + GRAV * EZ)
                + self.KP_ON * (p - pos)
                + self.KD_ON * (v - vel)
                - self.C_DRAG * vel
                - GRAV * EZ
            )
        self.u_hist = np.roll(self.u_hist, 1, axis=0)
        self.u_hist[0] = u_applied
        q = R.from_euler("z", self._yaw_at(t)).as_quat()
        return np.concatenate((p, v, a_ff, q, np.zeros(3))).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the clock."""
        self.tick += 1
        return self.finished

    def episode_callback(self):
        """Reset the clock."""
        self.tick = 0

    def render_callback(self, sim: Sim):
        """Draw the planned path."""
        try:
            from crazyflow.sim.visualize import draw_line, draw_points

            t = self.tick / self.freq
            ts = t + np.arange(0, 3.0, 0.05)
            P, _, _ = self._ref(ts)
            draw_line(sim, P, rgba=(0.0, 1.0, 0.0, 1.0))
            draw_points(sim, P[:1], rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        except Exception:
            pass
