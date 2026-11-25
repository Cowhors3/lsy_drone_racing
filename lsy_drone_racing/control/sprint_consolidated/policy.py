"""Shared policy inputs, network and action scaling for training and deployment."""

import numpy as np

CONTROL_PERIOD = 0.02
ACCELERATION_SCALE = 20.0


def rotation(q, xp):
    x, y, z, w = (q[..., i] for i in range(4))
    return xp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def features(obs, previous, xp):
    """Batched, scaled observations; no hidden environment state is used."""
    pos = obs["pos"]
    count = obs["gate_sequence"].shape[1]
    row = xp.arange(pos.shape[0])
    passed = obs["n_gates_passed"].astype(xp.int32)
    terms = [
        pos / 2,
        obs["vel"] / 4,
        rotation(obs["quat"], xp).reshape(-1, 9),
        obs["ang_vel"] / 15,
        previous[:, :3],
    ]
    for ahead in (0, 1):
        slot = xp.minimum(passed + ahead, count - 1)
        gate = obs["gate_sequence"][row, slot].astype(xp.int32)
        sign = obs["gate_sequence_direction"][row, slot]
        r = rotation(obs["gates_quat"][row, gate], xp)
        relative = obs["gates_pos"][row, gate] - pos
        terms += [
            relative / 2,
            r[..., :, 0] * sign[:, None],
            xp.einsum("bij,bi->bj", r, obs["vel"]) / 4,
            xp.einsum("bij,bi->bj", r, relative) / 2,
            obs["gates_visited"][row, gate, None].astype(pos.dtype),
        ]
    terms += [
        (obs["obstacles_pos"][..., :2] - pos[:, None, :2]).reshape(pos.shape[0], -1)
        / 2,
        obs["obstacles_visited"].astype(pos.dtype),
        (passed[:, None] == xp.arange(count + 1)[None]).astype(pos.dtype),
    ]
    acceleration = (obs["vel"] - 4 * previous[:, 3:6]) / CONTROL_PERIOD
    terms.append(acceleration / ACCELERATION_SCALE)
    return xp.clip(xp.concatenate(terms, axis=-1), -10, 10).astype(xp.float32)


def network(layers, x, xp):
    for weight, bias in layers[:-1]:
        x = xp.tanh(x @ weight + bias)
    weight, bias = layers[-1]
    return x @ weight + bias


def smooth_action(action, previous, xp):
    """Filter roll/pitch commands; thrust keeps its immediate response."""
    response = xp.asarray([0.65, 0.65, 1.0], dtype=action.dtype)
    return previous[:, :3] + response * (action - previous[:, :3])


def command(action, limits, xp):
    """Three normalized actions -> roll, pitch, fixed yaw, collective thrust [N]."""
    a = xp.clip(action, -1, 1)
    thrust = limits[0] + (a[..., 2] + 1) * 0.5 * (limits[1] - limits[0])
    return xp.stack(
        (limits[2] * a[..., 0], limits[2] * a[..., 1], xp.zeros_like(thrust), thrust),
        axis=-1,
    )


def save_policy(path, layers, limits):
    arrays = {
        "n_layers": len(layers),
        "limits": np.asarray(limits),
        "format_version": 1,
    }
    for i, (w, b) in enumerate(layers):
        arrays[f"w{i}"], arrays[f"b{i}"] = np.asarray(w), np.asarray(b)
    # Replace atomically so an interrupted save cannot corrupt the previous checkpoint.
    temporary = str(path) + ".tmp.npz"
    np.savez(temporary, **arrays)
    import os

    os.replace(temporary, path)


def load_policy(path):
    with np.load(path, allow_pickle=False) as data:
        layers = [
            (data[f"w{i}"].copy(), data[f"b{i}"].copy())
            for i in range(int(data["n_layers"]))
        ]
        return layers, data["limits"].copy()
