"""End-to-end reinforcement learning for drone racing.

* ``env.py``: training environment (drone model, track geometry, observation, reward).
* ``train.py``: PPO training loop, saves the policy weights to ``policy.npz``.

The controller that flies the trained policy is ``control/xji_e2e_racer.py``.
"""
