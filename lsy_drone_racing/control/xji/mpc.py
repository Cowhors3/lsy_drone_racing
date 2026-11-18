"""Unconstrained preview MPC on an identified acceleration response model.

The state interface of the drone is used as an acceleration interface: the commanded acceleration
is tracked by the onboard attitude loop with a noticeable lag. This lag is captured per axis by a
finite impulse response (FIR) model identified from flight data,

    a[k] = sum_i h[i] * u[k - i] + b,

where ``u`` is the commanded and ``a`` the realised acceleration and ``b`` a slowly varying bias.
Position and velocity follow by integrating ``a``. Because the model is linear and no inequality
constraints are used, the receding horizon problem has a closed-form solution whose gains are
computed once in the constructor. Solving the MPC online is a handful of dot products.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# FIR coefficients (50 Hz) of the commanded -> realised acceleration response.
# The horizontal axes are slow because the attitude has to tilt first, the vertical axis only has
# to wait for the motors.
FIR_XY = (
    0.0083, 0.0649, 0.1099, 0.1232, 0.1069, 0.0773, 0.0531, 0.0417, 0.0386, 0.0362, 0.0312, 0.0244,
    0.0176, 0.0125, 0.0111, 0.0153, 0.0233, 0.0268, 0.0161, 0.0, 0.0, 0.0, 0.0, 0.0187, 0.0065,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0048,
)  # fmt: skip
FIR_Z = (
    0.0, 0.0, 0.1445, 0.2863, 0.2608, 0.124, 0.0091, 0.0, 0.0, 0.0, 0.0125, 0.0439, 0.054, 0.0205,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)  # fmt: skip


class PreviewMPC:
    """Single-axis preview MPC with precomputed gains.

    The cost over the horizon ``n`` is

        qp * |p - p_ref|^2 + qv * |v - v_ref|^2 + ru * |u - a_ref|^2 + rdu * |du|^2.

    Only the first input of the optimal sequence is returned.
    """

    def __init__(
        self,
        fir: tuple[float, ...],
        horizon: int,
        dt: float,
        qp: float,
        qv: float,
        ru: float,
        rdu: float,
        n_history: int,
    ):
        """Precompute the feedback and feedforward gains.

        Args:
            fir: FIR coefficients of the acceleration response.
            horizon: Number of prediction steps.
            dt: Control period in seconds.
            qp: Position error weight.
            qv: Velocity error weight.
            ru: Weight on the deviation of the input from the reference acceleration.
            rdu: Weight on input increments.
            n_history: Number of past inputs kept as model state.
        """
        n, m = horizon, n_history
        h = np.asarray(fir, float)
        h = np.r_[h, np.zeros(max(0, m + 1 - len(h)))][: m + 1]
        # Realised acceleration = h_future @ future inputs + h_past @ past inputs.
        h_future, h_past = np.zeros((n, n)), np.zeros((n, m))
        for j in range(n):
            for i in range(m + 1):
                idx = j - i
                if idx >= 0:
                    h_future[j, idx] += h[i]
                elif -idx - 1 < m:
                    h_past[j, -idx - 1] += h[i]
        # Double integrator maps from accelerations to velocities / positions at steps 1..n.
        jj = np.arange(1, n + 1)[:, None]
        ll = np.arange(n)[None, :]
        c_vel = dt * (ll < jj)
        c_pos = dt**2 * np.where(ll < jj, jj - ll - 0.5, 0.0)
        g_pos, g_vel = c_pos @ h_future, c_vel @ h_future
        diff = np.eye(n) - np.eye(n, k=-1)
        hessian = qp * g_pos.T @ g_pos + qv * g_vel.T @ g_vel + ru * np.eye(n) + rdu * diff.T @ diff
        first_row = np.linalg.solve(hessian, np.eye(n))[0]
        self._k_pos = qp * first_row @ g_pos.T
        self._k_vel = qv * first_row @ g_vel.T
        self._k_acc = ru * first_row
        self._k_last = rdu * (first_row @ diff.T)[0]
        self._past_pos, self._past_vel = c_pos @ h_past, c_vel @ h_past
        self._bias_pos, self._bias_vel = c_pos.sum(axis=1), c_vel.sum(axis=1)
        self._t = dt * np.arange(1, n + 1)
        self.fir = h
        self.horizon = n

    def solve(
        self,
        pos: float,
        vel: float,
        u_history: NDArray,
        bias: float,
        pos_ref: NDArray,
        vel_ref: NDArray,
        acc_ref: NDArray,
    ) -> float:
        """Compute the next acceleration command.

        Args:
            pos: Current position.
            vel: Current velocity.
            u_history: Past inputs, most recent first.
            bias: Estimated acceleration bias.
            pos_ref: Reference positions at steps 1..n.
            vel_ref: Reference velocities at steps 1..n.
            acc_ref: Reference accelerations at steps 1..n.

        Returns:
            The acceleration command for the current step.
        """
        free_pos = pos + self._t * vel + self._past_pos @ u_history + self._bias_pos * bias
        free_vel = vel + self._past_vel @ u_history + self._bias_vel * bias
        return (
            self._k_pos @ (pos_ref - free_pos)
            + self._k_vel @ (vel_ref - free_vel)
            + self._k_acc @ acc_ref
            + self._k_last * u_history[0]
        )

    def predict_acc(self, u_history: NDArray) -> float:
        """Predict the currently realised acceleration from the past inputs."""
        return float(self.fir[: len(u_history)] @ u_history)

    @property
    def dc_gain(self) -> float:
        """Steady state gain of the response model."""
        return float(self.fir.sum())
