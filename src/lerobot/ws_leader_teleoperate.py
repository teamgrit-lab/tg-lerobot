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
Simple script to get teleoperation from a leader robot and send it to a websocket.

Example:

```shell
python -m lerobot.ws_leader_teleoperate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
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

# A set of connected websocket clients
clients = set()


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


async def handler(websocket, path, teleop: Teleoperator):
    """
    Handle websocket connections. Add new clients and remove them when they disconnect.
    """
    clients.add(websocket)
    logging.info(f"Client connected: {websocket.remote_address}. Total clients: {len(clients)}")
    try:
        # Keep the connection open
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)
        logging.info(f"Client disconnected: {websocket.remote_address}. Total clients: {len(clients)}")


async def broadcast_actions(teleop: Teleoperator):
    """
    Broadcast teleop actions to all connected clients.
    """
    while True:
        action = teleop.get_action()
        if action and clients:
            message = json.dumps(action, cls=NumpyEncoder)
            # Use asyncio.gather to send messages to all clients concurrently
            await asyncio.gather(
                *[client.send(message) for client in clients],
                return_exceptions=False,  # Exceptions will be propagated
            )
        # Adjust sleep time to control the rate of sending actions
        await asyncio.sleep(1 / 60)  # ~60 Hz


@dataclass
class WebsocketServerConfig:
    host: str = "localhost"
    port: int = 8765
    endpoint: str = "/ws/teleop"


@dataclass
class LeaderTeleoperateConfig:
    teleop: TeleoperatorConfig
    ws: WebsocketServerConfig = field(default_factory=WebsocketServerConfig)


async def main(cfg: LeaderTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    # Pass the teleop instance to the handler using a partial function
    handler_with_teleop = lambda ws, path: handler(ws, path, teleop)

    async with websockets.serve(handler_with_teleop, cfg.ws.host, cfg.ws.port):
        logging.info(f"Websocket server started at ws://{cfg.ws.host}:{cfg.ws.port}{cfg.ws.endpoint}")
        await broadcast_actions(teleop)

    teleop.disconnect()


@draccus.wrap()
def leader_teleoperate(cfg: LeaderTeleoperateConfig):
    try:
        asyncio.run(main(cfg))
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    leader_teleoperate()

