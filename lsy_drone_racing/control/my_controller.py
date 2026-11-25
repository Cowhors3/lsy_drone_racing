"""Deploy our own trained policy using only the course's allowed observation."""

import sys
from pathlib import Path

import numpy as np
from lsy_drone_racing.control import Controller

FOLDER = Path(__file__).resolve().parent / "sprint_consolidated"
if str(FOLDER) not in sys.path:
    sys.path.insert(0, str(FOLDER))
from policy import command, features, load_policy, network, smooth_action


class LearnedRacer(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("Use control_mode = 'attitude'")
        self.layers, self.limits = load_policy(FOLDER / "policy.npz")
        self.episode_callback()

    def compute_control(self, obs, info=None):
        if self.fresh:
            self.previous[:, 3:6] = np.asarray(obs["vel"])[None] / 4
            self.fresh = False
        batch = {key: np.asarray(value)[None] for key, value in obs.items()}
        inputs = features(batch, self.previous, np)
        action = smooth_action(
            np.tanh(network(self.layers, inputs, np)), self.previous, np
        )
        self.previous = np.concatenate(
            [action, np.asarray(obs["vel"])[None] / 4], axis=-1
        )
        return command(action, self.limits, np)[0].astype(np.float32)

    def episode_callback(self):
        self.previous = np.zeros((1, 6), dtype=np.float32)
        self.fresh = True

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return False
