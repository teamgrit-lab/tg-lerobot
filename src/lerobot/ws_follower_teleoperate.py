# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from a websocket teleoperation.

Example:

```shell
python -m lerobot.ws_follower_teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.host=localhost \
    --teleop.port=8765 \
    --teleop.endpoint=/ws/teleop
```
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import rerun as rr
import websockets

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class WSTeleoperatorConfig:
    """Configuration for the websocket teleoperator."""

    type: str = "ws"
    host: str = "localhost"
    port: int = 8765
    endpoint: str = "/ws/teleop"


class WSTeleoperator(Teleoperator):
    def __init__(self, cfg: WSTeleoperatorConfig):
        self.cfg = cfg
        self._thread = None
        self.latest_action = None
        self.websocket_url = f"ws://{self.cfg.host}:{self.cfg.port}{self.cfg.endpoint}"
        self.loop = None
        self.is_running = False

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_async())

    async def _main_async(self):
        while self.is_running:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logging.info(f"Connected to websocket server at {self.websocket_url}")
                    await self._recv_loop(websocket)
            except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
                logging.error(f"Failed to connect or connection lost: {e}. Retrying in 5 seconds.")
                await asyncio.sleep(5)

    async def _recv_loop(self, websocket):
        while self.is_running:
            try:
                message = await websocket.recv()
                # Assuming the message is received as bytes, decode it first.
                decoded_message = message.decode("utf-8")
                data = json.loads(decoded_message)
                self.latest_action = {k: np.array(v) for k, v in data.items()}
            except websockets.exceptions.ConnectionClosed:
                logging.warning("Websocket connection closed.")
                break  # This will trigger the reconnect logic in _main_async
            except json.JSONDecodeError:
                logging.warning("Received invalid JSON message")
            except Exception as e:
                logging.error(f"An error occurred in the receive loop: {e}")
                break

    def connect(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        logging.info("WSTeleoperator thread started.")

    def disconnect(self):
        self.is_running = False
        if self.loop:
            # This is not a perfect cleanup, as the loop might be sleeping.
            # But it's a daemon thread, so it will exit with the main program.
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread:
            self._thread.join(timeout=1)
        logging.info("WSTeleoperator thread stopped.")

    def get_action(self):
        return self.latest_action

    @property
    def is_connected(self):
        # This is a bit of a simplification. We assume if the thread is alive, it's trying to connect.
        return self._thread is not None and self._thread.is_alive()


@dataclass
class TeleoperateConfig:
    robot: RobotConfig
    teleop: WSTeleoperatorConfig = field(default_factory=WSTeleoperatorConfig)
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: WSTeleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    while not robot.is_connected:
        logging.info("Waiting for robot to connect...")
        time.sleep(1.0)

    # Wait for the robot to be ready and have action features
    while not robot.action_features:
        logging.info("Waiting for robot action features...")
        time.sleep(0.1)

    display_len = max(len(key) for key in robot.action_features)

    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()

        if action is None:
            # Wait for the first action to arrive
            logging.info("Waiting for first action from websocket...")
            time.sleep(0.5)
            continue

        if display_data:
            observation = robot.get_observation()
            log_rerun_data(observation, action)

        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        # Clear previous print output
        move_cursor_up(len(action) + 4)

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value[0]:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


def make_teleoperator_from_config(cfg: Any) -> Teleoperator:
    if cfg.type == "ws":
        return WSTeleoperator(cfg)
    raise ValueError(f"Unknown teleoperator type: {cfg.type}")


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()

