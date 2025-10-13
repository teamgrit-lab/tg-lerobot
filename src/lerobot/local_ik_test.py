"""
VR IK control script for robot teleoperation via websocket.

Examples:

테스트 모드 (더미 로봇):
```shell
python -m lerobot.ws_follower_ik \
    --robot.type=so101_follower \
    --test_mode=true \
    --ws.host=localhost \
    --ws.port=8765 \
    --ws.endpoint=/ws/vr_teleop
```

실제 로봇 모드:
```shell
python -m lerobot.ws_follower_ik \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --test_mode=false \
    --ws.host=localhost \
    --ws.port=8765 \
    --ws.endpoint=/ws/vr_teleop
```
"""

import asyncio
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any, Dict

import draccus
import numpy as np
from lerobot.ws_follower_ik import follower_ik_teleoperate
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
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from tests.rl.test_actor_learner import cfg

# 역기구학을 위한 간단한 클래스
class SimpleIK:
    """간단한 2DOF 역기구학 클래스"""
    
    def __init__(self, l1=0.2, l2=0.2):
        self.l1 = l1  # 첫 번째 링크 길이
        self.l2 = l2  # 두 번째 링크 길이
    
    def inverse_kinematics(self, x, y):
        """
        2DOF 역기구학 계산
        
        Args:
            x: 목표 x 좌표
            y: 목표 y 좌표
            
        Returns:
            tuple: (shoulder_lift, elbow_flex) 각도 (degrees)
        """
        # 목표점까지의 거리
        r = math.sqrt(x**2 + y**2)
        
        # 도달 가능 범위 확인
        if r > (self.l1 + self.l2) or r < abs(self.l1 - self.l2):
            # 도달 불가능한 경우 가장 가까운 점으로 조정
            if r > (self.l1 + self.l2):
                scale = (self.l1 + self.l2 - 0.01) / r
                x *= scale
                y *= scale
                r = math.sqrt(x**2 + y**2)
        
        # 코사인 법칙을 사용한 각도 계산
        cos_q2 = (r**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_q2 = max(-1, min(1, cos_q2))  # 범위 제한
        
        q2 = math.acos(cos_q2)  # elbow_flex (라디안)
        
        # 첫 번째 관절 각도
        alpha = math.atan2(y, x)
        beta = math.atan2(self.l2 * math.sin(q2), self.l1 + self.l2 * math.cos(q2))
        q1 = alpha - beta  # shoulder_lift (라디안)
        
        # 라디안을 도로 변환
        return math.degrees(q1), math.degrees(q2)
    
class IKMoveController:
    """로컬 IK 제어 (테스트용)"""
    
    def __init__(self, robot):
        self.robot = robot
        self.ik_solver = SimpleIK()
        
        # # 현재 목표 위치
        # self.current_target = {
        #     "shoulder_pan": 0.0,
        #     "shoulder_lift": 0.0, 
        #     "elbow_flex": 0.0,
        #     "wrist_flex": 0.0,
        #     "wrist_roll": 0.0,
        #     "gripper": 0.0,
        # }
        
        # P 제어 게인
        self.kp = 0.8

    def move_to(self, x: float, y: float):
        """목표 (x, y)로 이동"""
        shoulder_lift, elbow_flex = self.ik_solver.inverse_kinematics(x, y)
        logging.info(f"🎯 목표 위치: x={x:.3f}, y={y:.3f}")
        logging.info(f"🎯 목표 관절 각도: shoulder_lift={shoulder_lift:.2f}°, elbow_flex={elbow_flex:.2f}°")

        # 목표 관절값 구성
        target = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": shoulder_lift,
            "elbow_flex.pos": elbow_flex,
            "wrist_flex.pos": -shoulder_lift - elbow_flex,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }

        # 제어 루프
        for step in range(50):  # 50번 반복 (약 1초)
            obs = self.robot.get_observation()
            action = {}
            for key, target_val in target.items():
                current_val = obs.get(key, 0.0)
                error = target_val - current_val
                control = current_val + self.kp * error
                action[key] = control

            self.robot.send_action(action)
            time.sleep(0.02)

        logging.info("✅ 이동 완료")

@dataclass
class IKMoveConfig:
    x: float = 0.2  # x 좌표 (미터)
    y: float = 0.1  # y 좌표 (미터)
    test_mode: bool = True  # 테스트 모드 여부
    robot: RobotConfig = None

@draccus.wrap()
def main(cfg: IKMoveConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.test_mode:
        logging.info("🧪 테스트 모드 - 더미 로봇 사용")

        class DummyRobot:
            def __init__(self):
                self.is_connected = True

            def get_observation(self):
                return {
                    "shoulder_pan.pos": 0.0,
                    "shoulder_lift.pos": 0.0,
                    "elbow_flex.pos": 0.0,
                    "wrist_flex.pos": 0.0,
                    "wrist_roll.pos": 0.0,
                    "gripper.pos": 0.0,
                }

            def send_action(self, action):
                print("🤖 액션 전송:")
                for k, v in action.items():
                    print(f"   {k}: {v:.3f}")
                print("-" * 30)

            def disconnect(self):
                print("🔌 더미 로봇 연결 해제")

        robot = DummyRobot()

    else:
        logging.info("🤖 실제 로봇 연결 중...")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

    controller = IKMoveController(robot)
    controller.move_to(cfg.x, cfg.y)

    robot.disconnect()

if __name__ == "__main__":
    main()
