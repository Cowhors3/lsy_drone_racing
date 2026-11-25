"""Training environment for end-to-end drone racing, written in pure JAX.

The policy sees the drone state and the next gates, and outputs attitude commands directly. There
is no planner and no reference trajectory. Everything in this file is a pure function of a single
environment, so thousands of copies can run in parallel on the GPU with ``jax.vmap``.

The file is organised in the order the data flows:

1. Constants        sizes, action limits, reward weights, randomization ranges
2. Track            the nominal track loaded from a ``config/*.toml`` file
3. State            everything that changes during an episode
4. Dynamics         drone model (identified ``so_rpy_rotor_drag`` model of the Crazyflie)
5. Geometry         gate passing and collision checks
6. Observation      what the policy sees (shared with the deployed controller)
7. Reset and step   the usual RL interface
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import toml
from crazyflow.dynamics import load_params
from jax import Array
from jax.scipy.spatial.transform import Rotation

# region 1. Constants

CONTROL_DT = 1 / 50  # The environment and the policy run at 50 Hz
SUBSTEPS = 4  # Physics steps per control step
MAX_STEPS = 600  # Episode time limit (12 s)
TAKEOFF_STEPS = 25  # Touching the floor is allowed during the first 0.5 s

# Action: the policy outputs 3 numbers in [-1, 1] -> roll, pitch, collective thrust. Yaw stays 0.
ACTION_DIM = 3
MAX_TILT = 0.6  # rad, default roll and pitch limit. Train with ``--max_tilt`` for sharper turns
N_ACTION_HISTORY = 2  # Past actions in the observation, lets the policy cope with delay

# Geometry of the course hardware (see envs/assets/gate.xml and obstacle.xml)
# Drone approximated as a sphere. The simulator uses a 0.14 m wide box, so 0.10 leaves 3 cm of
# margin on each side. Train with ``--drone_radius`` to trade this margin for speed.
DRONE_RADIUS = 0.10
GATE_OPENING = 0.2  # Half width of the square opening
GATE_OUTER = 0.36  # Half width of the gate frame
GATE_DEPTH = 0.01  # Half thickness of the gate frame
STAND_HALF_WIDTH = 0.05  # The stand is a square post below the frame
POLE_RADIUS = 0.015
N_GATES_AHEAD = 2  # The policy sees the next two gates
AIM_MAX_DISTANCE = 0.6  # How far in front of a gate the reward may place its aim point

# Reward weights
R_PROGRESS = 1.0  # Per meter of approach to the next gate
R_GATE = 3.0  # Bonus for passing a gate
R_CRASH = -2.0
R_TIME = 0.01  # Default penalty per step. Train with ``--time_penalty`` to push for lap time
R_ACTION_RATE = 0.01  # Penalty on action changes, keeps commands smooth for the real drone
R_SPIN = 0.002  # Default penalty on roll and pitch rates, ``--spin_penalty`` allows quick turns
R_SPEED = 0.2  # Penalty on flying faster than MAX_SPEED
MAX_SPEED = 2.0  # m/s, default speed limit. Train with ``--max_speed`` for faster laps

# Domain randomization, makes the policy robust to the differences between simulation and the
# real drone. Each value is redrawn for every episode.
MASS_RANGE = 0.005  # kg, +-
THRUST_SCALE_RANGE = 0.15  # Default +-15 % thrust, covers battery level and motor wear
RPY_GAIN_RANGE = 0.20  # +-20 % strength of the onboard attitude response
RPY_BIAS_RANGE = 0.03  # rad, constant roll and pitch offset of the real drone
START_YAW_RANGE = 0.2  # rad, the drone is never placed perfectly straight
MAX_DELAY_STEPS = 2  # By default commands arrive 0, 1 or 2 control steps late (0 to 40 ms)
DELAY_BUFFER = 4  # Longest delay the environment can simulate, in control steps

# Measurement noise (standard deviation), redrawn at every step
POS_NOISE = 0.003  # m, motion capture
VEL_NOISE = 0.05  # m/s, the velocity is estimated and therefore noisier
RPY_NOISE = 0.01  # rad

OBS_DIM = 3 + 3 + 9 + 6 * N_GATES_AHEAD + 2 * 4 + ACTION_DIM * N_ACTION_HISTORY


# region 2. Track


class Track(NamedTuple):
    """Nominal track layout and the constants of the drone model."""

    gates_pos: Array  # (G, 3)
    gates_rpy: Array  # (G, 3)
    obstacles_pos: Array  # (O, 3), top of each pole
    start_pos: Array  # (3,)
    gate_sequence: Array  # (S,) gate ids in the order they must be passed
    gate_direction: Array  # (S,) +1 forward pass, -1 reverse pass
    sensor_range: Array  # True object positions are only visible within this distance
    pos_low: Array  # (3,) flight area limits
    pos_high: Array  # (3,)
    # Randomization half ranges, the same as the level uses in simulation
    gate_pos_noise: Array  # (3,)
    gate_rpy_noise: Array  # (3,)
    obstacle_noise: Array  # (3,)
    max_speed: Array  # m/s, flying faster than this is penalized
    time_penalty: Array  # Reward lost per step, the only term that rewards a short lap time
    drone_radius: Array  # m, collision size of the drone during training
    max_tilt: Array  # rad, roll and pitch command at the action limit
    spin_penalty: Array  # Reward lost per (rad/s)^2 of roll and pitch rate
    thrust_range: Array  # Thrust of the real drone differs by up to this fraction
    max_delay: Array  # Commands arrive up to this many control steps late
    # Drone model
    mass: Array
    thrust_low: Array  # N, total thrust limits
    thrust_high: Array
    thrust_coef: Array
    thrust_rate: Array
    drag: Array  # (3,) body frame drag coefficients
    rpy_coef: Array  # (3,)
    rpy_rates_coef: Array  # (3,)
    cmd_rpy_coef: Array  # (3,)


def load_track(
    config: str = "level2.toml",
    max_speed: float = MAX_SPEED,
    time_penalty: float = R_TIME,
    drone_radius: float = DRONE_RADIUS,
    max_tilt: float = MAX_TILT,
    spin_penalty: float = R_SPIN,
    thrust_range: float = THRUST_SCALE_RANGE,
    max_delay: int = MAX_DELAY_STEPS,
    drone: str = "cf21B_500",
) -> Track:
    """Load the nominal track and the drone parameters."""
    cfg = toml.load(Path(__file__).parents[3] / "config" / config)["env"]
    track, noise = cfg["track"], cfg["randomizations"]
    order = jnp.array(track["gate_order"])
    p = load_params("so_rpy_rotor_drag", drone)
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731
    return Track(
        gates_pos=f32([g["pos"] for g in track["gates"]]),
        gates_rpy=f32([g["rpy"] for g in track["gates"]]),
        obstacles_pos=f32([o["pos"] for o in track["obstacles"]]),
        start_pos=f32(track["drones"][0]["pos"]),
        gate_sequence=jnp.abs(order) - 1,
        gate_direction=jnp.sign(order),
        sensor_range=f32(cfg["sensor_range"]),
        pos_low=f32(track["safety_limits"]["pos_limit_low"]),
        pos_high=f32(track["safety_limits"]["pos_limit_high"]),
        gate_pos_noise=f32(noise["gate_pos"]["kwargs"]["maxval"]),
        gate_rpy_noise=f32(noise["gate_rpy"]["kwargs"]["maxval"]),
        obstacle_noise=f32(noise["obstacle_pos"]["kwargs"]["maxval"]),
        max_speed=f32(max_speed),
        time_penalty=f32(time_penalty),
        drone_radius=f32(drone_radius),
        max_tilt=f32(max_tilt),
        spin_penalty=f32(spin_penalty),
        thrust_range=f32(thrust_range),
        max_delay=jnp.asarray(min(max_delay, DELAY_BUFFER)),
        mass=f32(p["mass"]),
        thrust_low=f32(4 * p["thrust_min"]),
        thrust_high=f32(4 * p["thrust_max"]),
        thrust_coef=f32(p["cmd_f_coef"]),
        thrust_rate=f32(p["thrust_dyn_coef"]),
        drag=f32(p["drag_matrix"]).diagonal(),
        rpy_coef=f32(p["rpy_coef"]),
        rpy_rates_coef=f32(p["rpy_rates_coef"]),
        cmd_rpy_coef=f32(p["cmd_rpy_coef"]),
    )


# region 3. State


class State(NamedTuple):
    """Everything that changes during one episode of one environment."""

    # Drone
    pos: Array  # (3,)
    vel: Array  # (3,)
    rpy: Array  # (3,)
    rpy_rates: Array  # (3,)
    thrust: Array  # Current total thrust in N, lags behind the command
    actions: Array  # (1 + DELAY_BUFFER, 3) most recent policy actions, newest first
    # Race progress
    target: Array  # Index into the gate sequence of the next gate to pass
    step: Array
    # True (randomized) track of this episode and which objects the drone has seen so far
    gates_pos: Array  # (G, 3)
    gates_quat: Array  # (G, 4)
    obstacles_pos: Array  # (O, 3)
    gates_seen: Array  # (G,)
    obstacles_seen: Array  # (O,)
    # Randomized drone properties of this episode
    mass: Array
    thrust_scale: Array
    rpy_gain: Array
    rpy_bias: Array  # (3,)
    delay: Array  # Command delay in control steps
    key: Array


# region 4. Dynamics


def action_to_command(
    action: Array, thrust_low: Array, thrust_high: Array, max_tilt: Array | float
) -> Array:
    """Map a policy action in [-1, 1] to the attitude command [roll, pitch, yaw, thrust]."""
    action = jnp.clip(action, -1.0, 1.0)
    thrust = thrust_low + (action[2] + 1) / 2 * (thrust_high - thrust_low)
    return jnp.array([action[0] * max_tilt, action[1] * max_tilt, 0.0, thrust])


def simulate(state: State, command: Array, track: Track) -> State:
    """Advance the drone by one control step with the identified attitude-level model."""
    dt = CONTROL_DT / SUBSTEPS
    gravity = jnp.array([0.0, 0.0, -9.81])

    def substep(carry: tuple, _: None) -> tuple:
        pos, vel, rpy, rpy_rates, thrust = carry
        rot = Rotation.from_euler("xyz", rpy).as_matrix()
        # Translation: thrust along the body z axis, drag in the body frame, gravity
        force = state.thrust_scale * track.thrust_coef * thrust * rot[:, 2]
        force = force + rot @ (track.drag * (rot.T @ vel))
        vel = vel + (force / state.mass + gravity) * dt
        pos = pos + vel * dt
        # Floor: the drone rests on it until the motors have spun up
        on_floor = pos[2] <= 0.0
        pos = pos.at[2].set(jnp.maximum(pos[2], 0.0))
        vel = jnp.where(on_floor, jnp.zeros(3), vel)
        # Rotation: the onboard attitude controller behaves like a second order system
        target_rpy = command[:3] + state.rpy_bias
        rpy_acc = state.rpy_gain * (track.rpy_coef * rpy + track.cmd_rpy_coef * target_rpy)
        rpy_acc = rpy_acc + track.rpy_rates_coef * rpy_rates
        rpy_rates = rpy_rates + rpy_acc * dt
        rpy = rpy + rpy_rates * dt
        # Motors: thrust follows the command with a first order lag
        thrust = thrust + track.thrust_rate * (command[3] - thrust) * dt
        return (pos, vel, rpy, rpy_rates, thrust), None

    carry = (state.pos, state.vel, state.rpy, state.rpy_rates, state.thrust)
    (pos, vel, rpy, rpy_rates, thrust), _ = jax.lax.scan(substep, carry, None, SUBSTEPS)
    return state._replace(pos=pos, vel=vel, rpy=rpy, rpy_rates=rpy_rates, thrust=thrust)


# region 5. Geometry


def to_gate_frames(pos: Array, gates_pos: Array, gates_quat: Array) -> Array:
    """Express a world position in the frame of every gate. Gates are passed along local x."""
    return Rotation.from_quat(gates_quat).apply(pos - gates_pos, inverse=True)


def check_gates(last_pos: Array, pos: Array, state: State, track: Track) -> tuple[Array, Array]:
    """Check the motion of one step against all gates.

    Returns:
        passed: True if the drone flew through its target gate in the right direction.
        crashed: True if the drone touched any gate frame or stand, or backed out of the gate it
            has just passed.
    """
    last = to_gate_frames(last_pos, state.gates_pos, state.gates_quat)  # (G, 3)
    now = to_gate_frames(pos, state.gates_pos, state.gates_quat)
    # Where did the drone cross each gate plane? If it did not cross, use the current position.
    crossed = last[:, 0] * now[:, 0] < 0
    alpha = jnp.where(crossed, last[:, 0] / (last[:, 0] - now[:, 0]), 1.0)
    yz = last[:, 1:] + alpha[:, None] * (now[:, 1:] - last[:, 1:])
    offset = jnp.max(jnp.abs(yz), axis=-1)  # Distance from the gate center line, square norm

    in_opening = offset < GATE_OPENING - track.drone_radius
    in_frame = ~in_opening & (offset < GATE_OUTER + track.drone_radius)
    at_plane = crossed | (jnp.abs(now[:, 0]) < GATE_DEPTH + track.drone_radius)
    hit_frame = at_plane & in_frame
    reach = STAND_HALF_WIDTH + track.drone_radius
    hit_stand = (
        (jnp.abs(now[:, 0]) < reach) & (jnp.abs(now[:, 1]) < reach) & (now[:, 2] < -GATE_OUTER)
    )

    gate = track.gate_sequence[state.target]
    right_way = (now[gate, 0] - last[gate, 0]) * track.gate_direction[state.target] > 0
    passed = crossed[gate] & in_opening[gate] & right_way

    # A pass only needs the drone center to cross the gate plane. Without this rule the policy
    # learns to dip a few centimeters into a gate and reverse out of it when the next gate lies
    # behind it. Coming back through the opening of the gate just passed therefore ends the
    # episode, so the drone has to fly through completely and go around the frame.
    last_pass = jnp.maximum(state.target - 1, 0)
    last_gate = track.gate_sequence[last_pass]
    came_back = (now[last_gate, 0] - last[last_gate, 0]) * track.gate_direction[last_pass] < 0
    backed_out = crossed[last_gate] & in_opening[last_gate] & came_back
    backed_out &= (state.target > 0) & (last_gate != gate)  # The same gate may be the next target
    return passed, jnp.any(hit_frame | hit_stand) | backed_out


def check_obstacles(pos: Array, state: State, track: Track) -> Array:
    """True if the drone touches one of the vertical poles."""
    dist = jnp.linalg.norm(pos[:2] - state.obstacles_pos[:, :2], axis=-1)
    below_top = pos[2] < state.obstacles_pos[:, 2] + track.drone_radius
    return jnp.any((dist < POLE_RADIUS + track.drone_radius) & below_top)


def approach_distance(pos: Array, gate_pos: Array, gate_quat: Array, direction: Array) -> Array:
    """Distance to an aim point that guides the drone to the front of the gate.

    Flying straight at the gate center would pull a drone that is beside or behind the gate into
    the frame. The aim point therefore sits in front of the gate, further out the more the drone
    is off the gate axis, and moves into the gate center as the drone lines up.
    """
    local = Rotation.from_quat(gate_quat).apply(pos - gate_pos, inverse=True)
    along = local[0] * direction  # Negative in front of the gate, positive behind it
    off_axis = jnp.linalg.norm(local[1:]) + 2.0 * jnp.maximum(along, 0.0)
    normal = Rotation.from_quat(gate_quat).as_matrix()[:, 0] * direction
    aim = gate_pos - normal * jnp.minimum(off_axis, AIM_MAX_DISTANCE)
    return jnp.linalg.norm(aim - pos)


# region 6. Observation


def observe(
    pos: Array,
    vel: Array,
    rot: Array,
    gates_pos: Array,
    gates_quat: Array,
    obstacles_pos: Array,
    targets: Array,
    directions: Array,
    last_actions: Array,
) -> Array:
    """Build the policy input. Used both in training and by the deployed controller.

    Args:
        pos: Drone position, shape (3,).
        vel: Drone velocity, shape (3,).
        rot: Drone rotation matrix, shape (3, 3).
        gates_pos: Gate positions as currently known to the drone, shape (G, 3).
        gates_quat: Gate orientations as currently known to the drone, shape (G, 4).
        obstacles_pos: Pole positions as currently known to the drone, shape (O, 3).
        targets: Ids of the next gates to pass, shape (N_GATES_AHEAD,).
        directions: Passing direction of those gates (+1 or -1), shape (N_GATES_AHEAD,).
        last_actions: The most recent policy actions, newest first, shape (N_ACTION_HISTORY, 3).

    Returns:
        The observation vector of length ``OBS_DIM``.
    """
    gate_offset = gates_pos[targets] - pos  # Where the next gates are, relative to the drone
    gate_normal = Rotation.from_quat(gates_quat[targets]).as_matrix()[:, :, 0] * directions[:, None]
    obstacle_offset = obstacles_pos[:, :2] - pos[:2]  # Poles are vertical, height is irrelevant
    parts = [pos, vel, rot, gate_offset, gate_normal, obstacle_offset, last_actions]
    return jnp.concatenate([p.reshape(-1) for p in parts])


def observe_state(state: State, track: Track) -> Array:
    """Observation of a training state. Unseen objects are reported at their nominal pose."""
    nominal_quat = Rotation.from_euler("xyz", track.gates_rpy).as_quat()
    gates_pos = jnp.where(state.gates_seen[:, None], state.gates_pos, track.gates_pos)
    gates_quat = jnp.where(state.gates_seen[:, None], state.gates_quat, nominal_quat)
    obstacles_pos = jnp.where(
        state.obstacles_seen[:, None], state.obstacles_pos, track.obstacles_pos
    )
    # The drone state is measured with noise
    k_pos, k_vel, k_rpy = jax.random.split(jax.random.fold_in(state.key, 1), 3)
    pos = state.pos + POS_NOISE * jax.random.normal(k_pos, (3,))
    vel = state.vel + VEL_NOISE * jax.random.normal(k_vel, (3,))
    rpy = state.rpy + RPY_NOISE * jax.random.normal(k_rpy, (3,))
    # After the last gate there is no further target, so the last gate is simply repeated
    ahead = jnp.minimum(state.target + jnp.arange(N_GATES_AHEAD), track.gate_sequence.shape[0] - 1)
    return observe(
        pos,
        vel,
        Rotation.from_euler("xyz", rpy).as_matrix(),
        gates_pos,
        gates_quat,
        obstacles_pos,
        track.gate_sequence[ahead],
        track.gate_direction[ahead],
        state.actions[:N_ACTION_HISTORY],
    )


def update_seen(state: State, track: Track) -> State:
    """Objects within sensor range reveal their true position and stay revealed."""
    gate_dist = jnp.linalg.norm(state.gates_pos - state.pos, axis=-1)
    obstacle_dist = jnp.linalg.norm(state.obstacles_pos - state.pos, axis=-1)
    return state._replace(
        gates_seen=state.gates_seen | (gate_dist < track.sensor_range),
        obstacles_seen=state.obstacles_seen | (obstacle_dist < track.sensor_range),
    )


# region 7. Reset and step


def reset(key: Array, track: Track) -> State:
    """Start a new episode with a freshly randomized track and drone.

    Half of the episodes start on the start pad. The other half start in flight just behind a
    random gate, so that the later gates are practised as often as the first one.
    """
    keys = jax.random.split(key, 17)

    def uniform(k: Array, half_range: Array | float, shape: tuple = ()) -> Array:
        return jax.random.uniform(k, shape, minval=-1.0, maxval=1.0) * half_range

    # Randomized track
    gates_pos = track.gates_pos + uniform(keys[0], track.gate_pos_noise, track.gates_pos.shape)
    gates_rpy = track.gates_rpy + uniform(keys[1], track.gate_rpy_noise, track.gates_rpy.shape)
    gates_quat = Rotation.from_euler("xyz", gates_rpy).as_quat()
    obstacles_pos = track.obstacles_pos + uniform(
        keys[2], track.obstacle_noise, track.obstacles_pos.shape
    )

    # Start on the pad (target 0) or behind the gate that precedes a random target
    n_targets = track.gate_sequence.shape[0]
    target = jnp.where(
        jax.random.bernoulli(keys[3]), 0, jax.random.randint(keys[4], (), 1, n_targets)
    )
    pad_pos = track.start_pos + uniform(keys[5], 0.1, (3,)) * jnp.array([1.0, 1.0, 0.0])
    prev_gate = track.gate_sequence[target - 1]
    exit_dir = Rotation.from_quat(gates_quat[prev_gate]).as_matrix()[:, 0]
    exit_dir = exit_dir * track.gate_direction[target - 1]
    behind = 0.25 + uniform(keys[6], 0.15)  # Distance already flown past the previous gate
    air_pos = gates_pos[prev_gate] + behind * exit_dir + uniform(keys[7], 0.08, (3,))
    air_speed = (0.5 + uniform(keys[8], 0.5)) * track.max_speed
    air_vel = air_speed * exit_dir + uniform(keys[9], 0.5, (3,))
    on_pad = target == 0
    pos = jnp.where(on_pad, pad_pos, air_pos)
    vel = jnp.where(on_pad, jnp.zeros(3), air_vel)

    mass = track.mass + uniform(keys[10], MASS_RANGE)
    state = State(
        pos=pos,
        vel=vel,
        rpy=jnp.array([0.0, 0.0, 1.0]) * uniform(keys[14], START_YAW_RANGE),
        rpy_rates=jnp.zeros(3),
        thrust=jnp.where(on_pad, track.thrust_low, mass * 9.81),
        actions=jnp.zeros((1 + DELAY_BUFFER, ACTION_DIM)),
        target=target,
        step=jnp.zeros((), dtype=jnp.int32),
        gates_pos=gates_pos,
        gates_quat=gates_quat,
        obstacles_pos=obstacles_pos,
        gates_seen=jnp.zeros(track.gates_pos.shape[0], dtype=bool),
        obstacles_seen=jnp.zeros(track.obstacles_pos.shape[0], dtype=bool),
        mass=mass,
        thrust_scale=1.0 + uniform(keys[11], track.thrust_range),
        rpy_gain=1.0 + uniform(keys[15], RPY_GAIN_RANGE),
        rpy_bias=jnp.array([1.0, 1.0, 0.0]) * uniform(keys[16], RPY_BIAS_RANGE, (3,)),
        delay=jax.random.randint(keys[12], (), 0, track.max_delay + 1),
        key=keys[13],
    )
    return update_seen(state, track)


def step(state: State, action: Array, track: Track) -> tuple[State, Array, Array, dict]:
    """Apply one policy action.

    Returns:
        The next state (already reset if the episode ended), the reward, the done flag, and a dict
        with episode statistics for logging.
    """
    # 1) Move the drone. The command that reaches the drone may be a few steps old.
    action = jnp.clip(action, -1.0, 1.0)
    actions = jnp.concatenate([action[None], state.actions[:-1]])
    command = action_to_command(
        actions[state.delay], track.thrust_low, track.thrust_high, track.max_tilt
    )
    moved = simulate(state, command, track)

    # 2) Check what happened during this step
    passed, hit_gate = check_gates(state.pos, moved.pos, state, track)
    out_of_bounds = jnp.any((moved.pos < track.pos_low) | (moved.pos > track.pos_high))
    out_of_bounds |= (moved.pos[2] <= 0.0) & (state.step > TAKEOFF_STEPS)  # Landed outside takeoff
    crashed = hit_gate | check_obstacles(moved.pos, state, track) | out_of_bounds
    target = state.target + passed
    finished = target >= track.gate_sequence.shape[0]
    timeout = state.step + 1 >= MAX_STEPS
    done = crashed | finished | timeout

    # 3) Reward: approach the next gate from its front, pass it, stay smooth, do not crash
    gate = track.gate_sequence[state.target]
    direction = track.gate_direction[state.target]
    distance = partial(
        approach_distance,
        gate_pos=state.gates_pos[gate],
        gate_quat=state.gates_quat[gate],
        direction=direction,
    )
    progress = distance(state.pos) - distance(moved.pos)
    reward = (
        R_PROGRESS * progress
        + R_GATE * passed
        + R_CRASH * crashed
        - track.time_penalty
        - R_ACTION_RATE * jnp.sum((action - state.actions[0]) ** 2)
        - track.spin_penalty * jnp.sum(moved.rpy_rates[:2] ** 2)
        - R_SPEED * jnp.maximum(jnp.linalg.norm(moved.vel) - track.max_speed, 0.0) ** 2
    )

    # 4) Continue the episode, or start a new one if it ended
    key, reset_key = jax.random.split(state.key)
    moved = moved._replace(actions=actions, target=target, step=state.step + 1, key=key)
    moved = update_seen(moved, track)
    fresh = reset(reset_key, track)
    next_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, moved)
    info = {"passed": passed, "crashed": crashed, "finished": finished, "done": done}
    return next_state, reward, done, info
