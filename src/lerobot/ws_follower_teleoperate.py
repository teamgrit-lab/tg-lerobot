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
    --ws.host=localhost \
    --ws.port=8765 \
    --ws.endpoint=/ws/teleop
```

간소화된 실행(추천):

고정값을 설정 파일로 관리하면 한 줄로 실행할 수 있습니다.

```shell
python -m lerobot.ws_follower_teleoperate --config_path=configs/ws_follower_grit.yaml
```

필요 시 일부만 CLI로 덮어쓸 수 있습니다. 예:

```shell
python -m lerobot.ws_follower_teleoperate --config_path=configs/ws_follower_grit.yaml --ws.host=localhost
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
from pathlib import Path

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
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# Shared state between the websocket thread and the main robot control thread
latest_action: dict[str, Any] | None = None
action_lock = threading.Lock()


async def receive_actions_loop(websocket, robot: Robot, log_enabled: bool = False):
    """Receive actions from the websocket and send them to the robot immediately."""
    global latest_action
    async for message in websocket:
        try:
            # Assuming the message is received as bytes, decode it first.
            decoded_message = message.decode("utf-8")
            if log_enabled:
                print(f"Received message: {decoded_message}")
            data = json.loads(decoded_message)
            action = {k: np.array(v) for k, v in data.items()}
            robot.send_action(action)

            with action_lock:
                latest_action = action
        except json.JSONDecodeError:
            logging.warning("Received invalid JSON message")
        except Exception as e:
            logging.error(f"An error occurred in the receive loop: {e}")
            break


async def websocket_client(url, robot: Robot, log_enabled: bool = False):
    """Manages the websocket connection and reconnection."""
    while True:
        try:
            async with websockets.connect(url, ping_timeout=None) as websocket:
                logging.info(f"Connected to websocket server at {url}")
                await receive_actions_loop(websocket, robot, log_enabled)
        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
            logging.error(f"Failed to connect or connection lost: {e}. Retrying in 5 seconds.")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"An unexpected error occurred in websocket_client: {e}. Retrying in 5 seconds.")
            await asyncio.sleep(5)


def run_websocket_client_in_thread(url, robot: Robot, log_enabled: bool = False):
    """Runs the asyncio websocket client in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_client(url, robot, log_enabled))


def get_action_from_shared_state() -> dict[str, Any] | None:
    """Safely get the latest action from the shared state."""
    with action_lock:
        return latest_action


@dataclass
class WebsocketClientConfig:
    host: str = "localhost"
    port: int = 8765
    endpoint: str = "/ws/teleop"
    # 외부 파일에서 공용 설정을 불러오기 위한 선택적 경로
    path: str | None = None

    def __post_init__(self):
        # 공용 설정 파일이 지정된 경우, 기본값과 비교해 필요한 항목만 병합
        if self.path:
            try:
                resolved_path = Path(self.path).expanduser()
                if not resolved_path.is_absolute():
                    here = Path(__file__).resolve()
                    project_root = next((p for p in here.parents if (p / "pyproject.toml").exists()), None)
                    if project_root is not None:
                        resolved_path = project_root / resolved_path

                ws_from_file = draccus.parse(WebsocketClientConfig, str(resolved_path), args=[])
                defaults = WebsocketClientConfig()
                if self.host == defaults.host:
                    self.host = ws_from_file.host
                if self.port == defaults.port:
                    self.port = ws_from_file.port
                if self.endpoint == defaults.endpoint:
                    self.endpoint = ws_from_file.endpoint
            except Exception as e:
                logging.warning(f"Failed to load ws config from path '{self.path}': {e}")


@dataclass
class FollowerTeleoperateConfig:
    robot: RobotConfig
    ws: WebsocketClientConfig = field(default_factory=WebsocketClientConfig)
    # Limit the maximum frames per second.
    fps: int = 30
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # 수신 문자열 출력 여부 (True일 때만 로그 출력)
    log: bool = False


def follower_info_loop(
    robot: Robot, display_data: bool = False, duration: float | None = None, log_enabled: bool = False
):
    """The main loop for displaying robot info."""
    while not robot.is_connected:
        logging.info("Waiting for robot to connect...")
        time.sleep(1.0)

    # Wait for the robot to be ready and have action features
    while not robot.action_features:
        logging.info("Waiting for robot action features...")
        time.sleep(0.1)

    display_len = max(len(key) for key in robot.action_features)

    start_time = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = get_action_from_shared_state()

        if action is None:
            # Wait for the first action to arrive
            if log_enabled:
                logging.info("Waiting for first action from websocket...")
            time.sleep(0.01)
            continue

        if display_data:
            observation = robot.get_observation()
            log_rerun_data(observation, action)

        loop_s = time.perf_counter() - loop_start

        # print info, but dont block robot control
        # if len(action) > 0:
        #     move_cursor_up(len(action) + 4)

        # print("\n" + "-" * (display_len + 10))
        # print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        # for motor, value in action.items():
        #     print(f"{motor:<{display_len}} | {float(value):>7.2f}")
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start_time >= duration:
            break
        # update display at ~10hz
        time.sleep(0.1)


@draccus.wrap()
def follower_teleoperate(cfg: FollowerTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    websocket_url = f"ws://{cfg.ws.host}:{cfg.ws.port}{cfg.ws.endpoint}"

    # Start websocket client in a background daemon thread
    ws_thread = threading.Thread(
        target=run_websocket_client_in_thread, args=(websocket_url, robot, cfg.log), daemon=True
    )
    ws_thread.start()

    try:
        # The main thread can be used for other tasks, like displaying info
        follower_info_loop(
            robot,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            log_enabled=cfg.log,
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        robot.disconnect()
        # The websocket thread is a daemon, so it will exit automatically when the main thread exits.


if __name__ == "__main__":
    follower_teleoperate()

