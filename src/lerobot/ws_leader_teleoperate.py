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
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
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


async def send_actions_loop(teleop: Teleoperator, websocket):
    """
    Get actions from the leader and send them through the websocket.
    """
    logging.info("Starting to send actions.")
    while True:
        try:
            action = teleop.get_action()
            if action:
                message = json.dumps(action, cls=NumpyEncoder)
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


@dataclass
class LeaderTeleoperateConfig:
    teleop: TeleoperatorConfig
    ws: WebsocketClientConfig = field(default_factory=WebsocketClientConfig)


async def main(cfg: LeaderTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    websocket_url = f"ws://{cfg.ws.host}:{cfg.ws.port}{cfg.ws.endpoint}"

    while True:
        try:
            logging.info(f"Attempting to connect to {websocket_url}...")
            async with websockets.connect(websocket_url) as websocket:
                logging.info(f"Connected to websocket server at {websocket_url}")
                await send_actions_loop(teleop, websocket)
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
