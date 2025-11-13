# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
# obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simple script to get teleoperation from a leader robot and send it to a websocket server.

Example:

```shell
python -m lerobot.ws_leader_teleoperate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --ws.host=localhost \
    --ws.port=8765 \
    --ws.endpoint=/ws/teleop
```

간소화된 실행(추천):

고정값을 설정 파일로 관리하면 한 줄로 실행할 수 있습니다.

```shell
python -m lerobot.ws_leader_teleoperate --config_path=configs/ws_leader_grit.yaml
```

필요 시 일부만 CLI로 덮어쓸 수 있습니다. 예:

```shell
python -m lerobot.ws_leader_teleoperate --config_path=configs/ws_leader_grit.yaml --ws.host=localhost
```
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import websockets

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.utils import init_logging


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


async def send_actions_loop(teleop: Teleoperator, websocket, log_enabled: bool = False):
    """
    Get actions from the leader and send them through the websocket.
    """
    logging.info("Starting to send actions.")
    while True:
        try:
            action = teleop.get_action()
            if action:
                message = json.dumps(action, cls=NumpyEncoder)
                if log_enabled:
                    print(f"Sending message: {message}")
                await websocket.send(message.encode("utf-8"))
            # Adjust sleep time to control the rate of sending actions
            await asyncio.sleep(1 / 30)  # ~30 Hz
        except websockets.exceptions.ConnectionClosed:
            logging.warning("Connection closed while sending. Will attempt to reconnect.")
            break
        except Exception as e:
            logging.error(f"An error occurred in send_actions_loop: {e}")
            break


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
                # 공용 파일은 루트에 host/port/endpoint 키를 가지는 YAML을 기대함
                # 상대경로인 경우, 프로젝트 루트(pyproject.toml이 있는 디렉토리) 기준으로 해석
                resolved_path = Path(self.path).expanduser()
                if not resolved_path.is_absolute():
                    here = Path(__file__).resolve()
                    project_root = next((p for p in here.parents if (p / "pyproject.toml").exists()), None)
                    if project_root is not None:
                        resolved_path = project_root / resolved_path

                ws_from_file = draccus.parse(WebsocketClientConfig, str(resolved_path), args=[])
                defaults = WebsocketClientConfig()
                # 사용자가 YAML에서 명시하지 않은 값(=기본값 유지)만 공용 파일 값으로 채운다
                if self.host == defaults.host:
                    self.host = ws_from_file.host
                if self.port == defaults.port:
                    self.port = ws_from_file.port
                if self.endpoint == defaults.endpoint:
                    self.endpoint = ws_from_file.endpoint
            except Exception as e:
                logging.warning(f"Failed to load ws config from path '{self.path}': {e}")


@dataclass
class LeaderTeleoperateConfig:
    teleop: TeleoperatorConfig
    ws: WebsocketClientConfig = field(default_factory=WebsocketClientConfig)
    # 전송 문자열 출력 여부 (True일 때만 93줄 로그 출력)
    log: bool = False


async def main(cfg: LeaderTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    websocket_url = f"ws://{cfg.ws.host}:{cfg.ws.port}{cfg.ws.endpoint}"

    while True:
        try:
            logging.info(f"Attempting to connect to {websocket_url}...")
            async with websockets.connect(websocket_url, ping_timeout=None) as websocket:
                logging.info(f"Connected to websocket server at {websocket_url}")
                await send_actions_loop(teleop, websocket, cfg.log)
        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
            logging.error(f"Failed to connect or connection lost: {e}. Retrying in 5 seconds.")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"An unexpected error occurred in main loop: {e}. Retrying in 5 seconds.")
            await asyncio.sleep(5)


@draccus.wrap()
def leader_teleoperate(cfg: LeaderTeleoperateConfig):
    try:
        asyncio.run(main(cfg))
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    leader_teleoperate()
