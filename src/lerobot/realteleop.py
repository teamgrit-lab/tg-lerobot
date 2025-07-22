
import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower_end_effector import (
    SO101FollowerEndEffectorConfig,
)
from lerobot.utils.utils import init_logging


@dataclass
class RealTeleopConfig:
    robot: SO101FollowerEndEffectorConfig
    fps: int = 10


def teleop_loop(robot, fps):
    display_len = 20  # For pretty printing
    while True:
        try:
            # Get user input for delta x, y, z
            delta_x_str = input("Enter delta x (float): ")
            delta_y_str = input("Enter delta y (float): ")
            delta_z_str = input("Enter delta z (float): ")
            gripper_str = input("Enter gripper (0 for open, 1 for close): ")

            # Convert to float
            delta_x = float(delta_x_str)
            delta_y = float(delta_y_str)
            delta_z = float(delta_z_str)
            gripper = float(gripper_str)

            action = {
                "delta_x": delta_x,
                "delta_y": delta_y,
                "delta_z": delta_z,
                "gripper": gripper,
            }

            loop_start = time.perf_counter()
            robot.send_action(action)
            loop_s = time.perf_counter() - loop_start

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'VALUE':>7}")
            for key, value in action.items():
                print(f"{key:<{display_len}} | {value:>7.2f}")
            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s if loop_s > 0 else 0:.0f} Hz)")

        except ValueError:
            print("Invalid input. Please enter a float.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        time.sleep(1 / fps)


@draccus.wrap()
def main(cfg: RealTeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        teleop_loop(robot, cfg.fps)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    # sensible defaults for SO101
    cfg = RealTeleopConfig(
        robot=SO101FollowerEndEffectorConfig(
            type="so101_follower_end_effector",
            port="/dev/tty.usbmodem58760431541",
            urdf_path="so101_new_calib.urdf",
        )
    )
    main(cfg) 