#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from ..config import RobotConfig
from .config_so101_follower import SO101FollowerConfig


@RobotConfig.register_subclass("so101_follower_end_effector")
@dataclass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig):
    urdf_path: str | None = None
    # name of the end-effector frame in the URDF
    target_frame_name: str = "gripper_frame_link"
    # min and max bounds for the end-effector position
    end_effector_bounds: dict = field(
        default_factory=lambda: {
            "min": [-0.5, -0.5, 0.0],
            "max": [0.5, 0.5, 0.5],
        }
    )
    # step sizes for the end-effector position
    end_effector_step_sizes: dict = field(
        default_factory=lambda: {
            "x": 0.01,
            "y": 0.01,
            "z": 0.01,
        }
    )
    max_gripper_pos: float = 150.0 