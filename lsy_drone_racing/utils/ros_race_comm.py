"""ROS2 node for the communication between host and clients in multi-drone races."""

from __future__ import annotations

import logging
import threading

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile

logger = logging.getLogger(__name__)

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RaceCommNode:
    """ROS2 node for race coordination, spinning in a background daemon thread.

    Access the underlying rclpy node via :attr:`node` to create publishers and
    subscriptions directly. All cleanup is handled by :meth:`close`.

    Args:
        name: ROS2 node name (must be unique within the process).
        use_sim_time: Whether the node reads its time from the ``/clock`` topic instead of the
            system clock.
    """

    def __init__(self, name: str, *, use_sim_time: bool = False):
        """Initialize and spin the ROS2 node in a background thread."""
        use_sim_time_parameter = Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)
        self.node = rclpy.create_node(name, parameter_overrides=[use_sim_time_parameter])
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, daemon=True, name=f"spin-{name}")
        self._thread.start()
        logger.debug(f"RaceCommNode '{name}' started")

    def _spin(self):
        """Spin the ROS2 node in a background thread."""
        try:
            self._executor.spin()
        except ExternalShutdownException:
            logger.debug(f"'{self.node.get_name()}' spin thread stopped")
        except _rclpy.RCLError:
            # Raised when the context is shut down while the thread is still spinning
            logger.debug(f"'{self.node.get_name()}' spin thread stopped (context invalid)")

    def close(self):
        """Shut down the executor and destroy the node."""
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        self._thread.join(timeout=1.0)
        logger.debug("RaceCommNode closed")
