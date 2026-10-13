"""Deployment script for multi-drone racing clients.

Clients run isolated and communicate with the race host via Zenoh. Make sure the host is up before
starting your client.

Usage:

    pixi run -e deploy python scripts/multi_deploy_client.py --config multi_level2.toml --rank 0
    pixi run -e deploy python scripts/multi_deploy_client.py --config multi_level2.toml --rank 1
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import extract_config_for_rank, load_config, load_controller

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_client_env import RealMultiDroneRaceEnvClient

logger = logging.getLogger(__name__)


def configure_logging(rank: int):
    """Configure logging for the client.

    Args:
        rank: Rank of this drone in the multi-drone setup.
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[Rank {rank}][%(asctime)s.%(msecs)03d] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def main(rank: int, config: str = "multi_level2.toml", controller: str | None = None):
    """Deploy and run a client for multi-drone racing.

    Args:
        rank: Rank of this drone in the multi-drone setup.
        config: Configuration file name, assumed to be in ``config/``.
        controller: Controller file name in ``lsy_drone_racing/control/``, or ``None`` to use the
            value from the config file.
    """
    configure_logging(rank)
    rclpy.init()
    config_obj = load_config(Path(__file__).parents[1] / "config" / config)
    if controller is not None:
        config_obj.controller[rank]["file"] = controller
    env: RealMultiDroneRaceEnvClient = gymnasium.make(
        "RealMultiDroneRaceEnvClient-v0",
        drones=config_obj.deploy.drones,
        rank=rank,
        freq=config_obj.env.kwargs[rank]["freq"],
        track=config_obj.env.track,
        randomizations=config_obj.env.randomizations,
        dynamics=config_obj.sim.dynamics,
        sensor_range=config_obj.env.kwargs[rank]["sensor_range"],
        control_mode=config_obj.env.kwargs[rank]["control_mode"],
    )
    try:
        obs, info = env.reset(options=config_obj.deploy)

        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        controller_cls = load_controller(control_path / config_obj.controller[rank]["file"])
        controller = controller_cls(obs, info, extract_config_for_rank(config_obj, rank))

        env.unwrapped.wait_for_race_start(timeout=120.0)
        logger.info("Started race")
        start_time = time.perf_counter()

        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                logger.debug(f"Race finished. {terminated=}, {truncated=}, {controller_finished=}")
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.kwargs[rank].freq):
                time.sleep(1 / config.env.kwargs[rank].freq - dt)
            else:
                exc = dt - 1 / config.env.kwargs[rank].freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3e}s.")

        ep_time = time.perf_counter() - start_time
        finished = bool(info["finished_track"])
        logger.info(f"Track time: {ep_time:.3f}s" if finished else "Task not completed")
        logger.info(f"Number of gates passed: {info['n_gates_passed']}")
    except KeyboardInterrupt:
        logger.info("Run aborted by user")
    finally:
        env.close()
        rclpy.shutdown()
        logger.debug("Shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
