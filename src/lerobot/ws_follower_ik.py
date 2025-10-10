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

# VR 제어를 위한 클래스
class VRIKController:
    """VR 입력을 사용한 IK 기반 로봇 제어"""
    
    def __init__(self, robot):
        self.robot = robot
        self.ik_solver = SimpleIK()
        
        # VR 좌표계를 로봇 좌표계로 변환하는 스케일
        self.vr_to_robot_scale = 0.5
        
        # 제어 상태
        self.origin_position = None
        self.origin_quaternion = None
        self.is_active = False
        
        # 그리퍼 상태
        self.gripper_closed = False
        
        # 현재 목표 위치
        self.current_target = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0, 
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # P 제어 게인
        self.kp = 0.8
        
    def quaternion_to_euler(self, quat):
        """쿼터니언을 오일러 각도로 변환"""
        x, y, z, w = quat
        
        # Roll (x축 회전)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y축 회전)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z축 회전)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def reset_to_zero_position(self):
        """로봇을 원점(제로 포지션)으로 리셋"""
        # 모든 관절을 0도로 설정
        self.current_target = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0, 
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # VR 원점도 리셋
        self.origin_position = None
        self.origin_quaternion = None
        self.is_active = False
        
        logging.info("🎯 로봇이 원점으로 리셋되었습니다")
    
    def process_vr_data(self, vr_data: Dict):
        """VR 데이터 처리 및 로봇 제어"""
        try:
            position = vr_data.get('position', {})
            orientation = vr_data.get('orientation', {})
            trigger = vr_data.get('trigger', 0)
            reset = vr_data.get('reset', False)
            
            # Reset 명령 처리
            if reset:
                self.reset_to_zero_position()
                logging.info("🔄 VR Reset 명령 수신 - 원점으로 이동")
                return
            
            # 위치 데이터 확인
            if not all(k in position for k in ['x', 'y', 'z']):
                return
            
            # VR 위치 추출
            vr_pos = np.array([
                float(position['x']),
                float(position['y']), 
                float(position['z'])
            ])
            
            # 원점 설정 (첫 번째 프레임)
            if self.origin_position is None:
                self.origin_position = vr_pos.copy()
                if all(k in orientation for k in ['x', 'y', 'z', 'w']):
                    self.origin_quaternion = np.array([
                        float(orientation['x']),
                        float(orientation['y']),
                        float(orientation['z']),
                        float(orientation['w'])
                    ])
                self.is_active = True
                logging.info("🎯 VR 원점 설정 완료 - 제어 활성화")
                return
            
            # 상대 위치 계산
            relative_pos = (vr_pos - self.origin_position) * self.vr_to_robot_scale
            
            # 로봇 좌표계로 변환 (VR Z -> 로봇 X, VR Y -> 로봇 Y)
            robot_x = 0.16 + relative_pos[2]  # 기본 위치에서 Z 변화량 적용
            robot_y = 0.11 + relative_pos[1]  # 기본 위치에서 Y 변화량 적용
            
            # 역기구학 계산
            try:
                shoulder_lift, elbow_flex = self.ik_solver.inverse_kinematics(robot_x, robot_y)
                
                # 부드러운 전환을 위한 필터링
                alpha = 0.1  # 필터링 계수 (0-1, 낮을수록 부드러움)
                self.current_target["shoulder_lift"] = (
                    (1 - alpha) * self.current_target["shoulder_lift"] + 
                    alpha * shoulder_lift
                )
                self.current_target["elbow_flex"] = (
                    (1 - alpha) * self.current_target["elbow_flex"] + 
                    alpha * elbow_flex
                )
                
            except Exception as e:
                logging.warning(f"IK 계산 실패: {e}")
                return
            
            # 손목 각도 처리
            if all(k in orientation for k in ['x', 'y', 'z', 'w']):
                current_quat = np.array([
                    float(orientation['x']),
                    float(orientation['y']),
                    float(orientation['z']),
                    float(orientation['w'])
                ])
                
                # 오일러 각도로 변환
                roll, pitch, yaw = self.quaternion_to_euler(current_quat)
                
                # 원점 대비 상대 각도 계산
                if self.origin_quaternion is not None:
                    origin_roll, origin_pitch, origin_yaw = self.quaternion_to_euler(self.origin_quaternion)
                    
                    # 상대 각도 (도 단위)
                    relative_roll = math.degrees(roll - origin_roll)
                    relative_pitch = math.degrees(pitch - origin_pitch)
                    
                    # 각도 제한
                    relative_roll = max(-90, min(90, relative_roll))
                    relative_pitch = max(-90, min(90, relative_pitch))
                    
                    self.current_target["wrist_roll"] = relative_roll
                    
                    # 손목 굽힘 각도 계산 (end-effector 방향 유지)
                    self.current_target["wrist_flex"] = (
                        -self.current_target["shoulder_lift"] - 
                        self.current_target["elbow_flex"] + 
                        relative_pitch
                    )
            
            # VR X축 변화량으로 shoulder_pan 제어
            pan_scale = 100.0
            delta_pan = relative_pos[0] * pan_scale
            delta_pan = max(-90, min(90, delta_pan))
            self.current_target["shoulder_pan"] = delta_pan
            
            # 그리퍼 제어
            if trigger > 0.5:
                self.current_target["gripper"] = 45.0  # 닫힘
            else:
                self.current_target["gripper"] = 0.0   # 열림
                
        except Exception as e:
            logging.error(f"VR 데이터 처리 중 오류: {e}")
    
    def get_control_action(self) -> Dict[str, float]:
        """현재 목표에 기반한 제어 액션 생성"""
        if not self.is_active:
            return {}
        
        try:
            obs = self.robot.get_observation()
            action = {}
            
            # 각 관절에 대한 P 제어
            joint_mapping = {
                "shoulder_pan": "shoulder_pan",
                "shoulder_lift": "shoulder_lift", 
                "elbow_flex": "elbow_flex",
                "wrist_flex": "wrist_flex",
                "wrist_roll": "wrist_roll",
                "gripper": "gripper"
            }
            
            for joint_name, obs_key in joint_mapping.items():
                if joint_name in self.current_target:
                    # 현재 위치 가져오기
                    current_pos = obs.get(f"{obs_key}.pos", 0.0)
                    
                    # 목표 위치
                    target_pos = self.current_target[joint_name]
                    
                    # P 제어
                    error = target_pos - current_pos
                    control = self.kp * error
                    
                    # 액션 설정
                    action[f"{obs_key}.pos"] = current_pos + control
            
            return action
            
        except Exception as e:
            logging.error(f"제어 액션 생성 중 오류: {e}")
            return {}

# 공유 상태
latest_vr_data: Dict[str, Any] | None = None
vr_data_lock = threading.Lock()
vr_controller: VRIKController | None = None

async def receive_vr_data_loop(websocket, robot):
    """웹소켓에서 VR 데이터 수신 및 처리"""
    global latest_vr_data, vr_controller
    
    async for message in websocket:
        try:
            # 메시지 디코딩
            if isinstance(message, bytes):
                decoded_message = message.decode("utf-8")
            else:
                decoded_message = message
                
            # JSON 파싱
            vr_data = json.loads(decoded_message)
            
            # VR 컨트롤러로 데이터 처리
            if vr_controller:
                vr_controller.process_vr_data(vr_data)
            
            # 공유 상태 업데이트
            with vr_data_lock:
                latest_vr_data = vr_data
                
        except json.JSONDecodeError:
            logging.warning("잘못된 JSON 메시지 수신")
        except Exception as e:
            logging.error(f"VR 데이터 수신 루프에서 오류 발생: {e}")
            break

async def websocket_client(url, robot):
    """웹소켓 클라이언트 관리 및 재연결"""
    while True:
        try:
            async with websockets.connect(url, ping_timeout=None) as websocket:
                logging.info(f"웹소켓 서버에 연결됨: {url}")
                await receive_vr_data_loop(websocket, robot)
        except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError, OSError) as e:
            logging.error(f"연결 실패 또는 연결 끊김: {e}. 5초 후 재시도.")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"웹소켓 클라이언트에서 예상치 못한 오류: {e}. 5초 후 재시도.")
            await asyncio.sleep(5)

def run_websocket_client_in_thread(url, robot):
    """별도 스레드에서 asyncio 웹소켓 클라이언트 실행"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_client(url, robot))

def get_vr_data_from_shared_state() -> Dict[str, Any] | None:
    """공유 상태에서 최신 VR 데이터 안전하게 가져오기"""
    with vr_data_lock:
        return latest_vr_data

@dataclass
class WebsocketClientConfig:
    host: str = "localhost"
    port: int = 8765
    endpoint: str = "/ws/vr_teleop"

@dataclass
class FollowerIKTeleoperateConfig:
    robot: RobotConfig
    ws: WebsocketClientConfig = field(default_factory=WebsocketClientConfig)
    # 최대 프레임 속도 제한
    fps: int = 30
    teleop_time_s: float | None = None
    # 모든 카메라를 화면에 표시
    display_data: bool = False
    # 테스트 모드 (True: 더미 로봇 사용, False: 실제 로봇 연결)
    test_mode: bool = True

def follower_ik_control_loop(robot, display_data: bool = False, duration: float | None = None):
    """VR IK 제어를 위한 메인 루프"""
    global vr_controller
    
    while not robot.is_connected:
        logging.info("로봇 연결 대기 중...")
        time.sleep(1.0)

    # VR 컨트롤러 초기화
    vr_controller = VRIKController(robot)
    logging.info("VR IK 컨트롤러 초기화 완료")

    start_time = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        
        # VR 데이터 확인
        vr_data = get_vr_data_from_shared_state()
        
        if vr_data is None:
            logging.info("첫 번째 VR 데이터 대기 중...")
            time.sleep(0.01)
            continue

        # 제어 액션 생성 및 전송
        if vr_controller and vr_controller.is_active:
            action = vr_controller.get_control_action()
            if action:
                robot.send_action(action)

        # 데이터 표시 (선택사항)
        if display_data and vr_controller and vr_controller.is_active:
            try:
                observation = robot.get_observation()
                # 간단한 액션 딕셔너리 생성 (rerun 로깅용)
                simple_action = {k: np.array([v]) for k, v in action.items()}
                log_rerun_data(observation, simple_action)
            except Exception as e:
                logging.warning(f"데이터 표시 중 오류: {e}")

        loop_s = time.perf_counter() - loop_start

        # 시간 제한 확인
        if duration is not None and time.perf_counter() - start_time >= duration:
            break
            
        # 30Hz로 업데이트
        time.sleep(max(0, 1.0/30 - loop_s))

@draccus.wrap()
def follower_ik_teleoperate(cfg: FollowerIKTeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        _init_rerun(session_name="vr_ik_teleoperation")

    # 테스트 모드 vs 실제 로봇 모드 선택
    if cfg.test_mode:
        logging.info("🧪 테스트 모드 - 더미 로봇 사용")
        
        # 테스트용 더미 로봇 객체
        class DummyRobot:
            def __init__(self):
                self.is_connected = True
                
            def get_observation(self):
                # 더미 관찰값 반환
                return {
                    "shoulder_pan.pos": 0.0,
                    "shoulder_lift.pos": 0.0,
                    "elbow_flex.pos": 0.0,
                    "wrist_flex.pos": 0.0,
                    "wrist_roll.pos": 0.0,
                    "gripper.pos": 0.0,
                }
                
            def send_action(self, action):
                # 액션 출력 (테스트용) - 더 읽기 쉽게 포맷팅
                if action:
                    print("🤖 로봇 액션:")
                    for joint, value in action.items():
                        print(f"   {joint}: {value:.3f}")
                    print("-" * 40)
                else:
                    print("🤖 로봇 액션: 없음")
                
            def disconnect(self):
                print("🔌 더미 로봇 연결 해제")
        
        robot = DummyRobot()
        
    else:
        logging.info("🤖 실제 로봇 모드 - 로봇에 연결")
        # 실제 로봇 연결
        robot = make_robot_from_config(cfg.robot)
        robot.connect()

    websocket_url = f"ws://{cfg.ws.host}:{cfg.ws.port}{cfg.ws.endpoint}"

    # 백그라운드 데몬 스레드에서 웹소켓 클라이언트 시작
    ws_thread = threading.Thread(
        target=run_websocket_client_in_thread, args=(websocket_url, robot), daemon=True
    )
    ws_thread.start()

    try:
        # 메인 스레드는 VR IK 제어 루프 실행
        follower_ik_control_loop(robot, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        robot.disconnect()
        # 웹소켓 스레드는 데몬이므로 메인 스레드 종료 시 자동으로 종료됨

if __name__ == "__main__":
    follower_ik_teleoperate()
