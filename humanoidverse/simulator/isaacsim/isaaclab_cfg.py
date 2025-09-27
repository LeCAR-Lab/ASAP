# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_CFG

# from humanoidverse.envs.base_task.events import OmniH2OEventCfg, OmniH2OPlayEventCfg, OmniH2OTrainEventCfg
# from humanoidverse.envs.base_task.modes import OmniH2OModes
# from humanoidverse.envs.base_task.rewards import RewardCfg

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../../../tests/data/")


@configclass
class IsaacLabCfg(DirectRLEnvCfg):
    # mode = OmniH2OModes.TRAIN

    # env
    episode_length_s = 3600.0
    substeps = 1
    decimation = 4
    action_scale = 0.25

    num_actions = 19
    num_observations = 913
    observation_space = 913
    action_space = 19
    num_self_obs = 342
    num_ref_obs = 552
    num_action_obs = 19

    num_states = 990

    dt = 0.005

    # If we are doing distill
    distill = False
    single_history_dim = 63
    short_history_length = 25
    distill_teleop_selected_keypoints_names = None

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=substeps, physx=PhysxCfg(bounce_threshold_velocity=0.2))
    # TODO(rhua): using flat terrain until RayCaster is fixed
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True)

    # robot
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit={
                ".*_hip_yaw": 200.0,
                ".*_hip_roll": 200.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 300.0,
                "torso": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw": 23.0,
                ".*_hip_roll": 23.0,
                ".*_hip_pitch": 23.0,
                ".*_knee": 14.0,
                "torso": 23.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit=40,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 18.0,
                ".*_elbow": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch": 9.0,
                ".*_shoulder_roll": 9.0,
                ".*_shoulder_yaw": 20.0,
                ".*_elbow": 20.0,
            },
            stiffness=0,
            damping=0,
        ),
    }

    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    joint_names = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
        "torso",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]

    base_name = "torso_link"

    feet_name = ".*_ankle_link"
    knee_name = ".*_knee_link"
    # extend links (these ids are ids after reordered, ie. these id are from IsaacGym, TODO: change to new id directly)
    # h1
    extend_body_parent_ids = [15, 19, 0]
    extend_body_pos = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [0, 0, 0.75]])

    teleop_selected_keypoints_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    # control parameters
    stiffness = {
        ".*_hip_yaw": 200.0,
        ".*_hip_roll": 200.0,
        ".*_hip_pitch": 200.0,
        ".*_knee": 300.0,
        ".*_ankle": 40.0,
        ".*_shoulder_pitch": 100.0,
        ".*_shoulder_roll": 100.0,
        ".*_shoulder_yaw": 100.0,
        ".*_elbow": 100.0,
        "torso": 300.0,
    }

    damping = {
        ".*_hip_yaw": 5.0,
        ".*_hip_roll": 5.0,
        ".*_hip_pitch": 5.0,
        ".*_knee": 6.0,
        ".*_ankle": 2.0,
        ".*_shoulder_pitch": 2.0,
        ".*_shoulder_roll": 2.0,
        ".*_shoulder_yaw": 2.0,
        ".*_elbow": 2.0,
        "torso": 6.0,
    }

    # control type: the action type from the policy
    # "Pos": target joint pos, "Vel": target joint vel, "Torque": joint torques
    control_type = "Pos"

    # Control delay step range (min, max): the control will be randomly delayed at least "min" steps and at most
    # "max" steps. If (0,0), then it means no delay happen
    ctrl_delay_step_range = (0, 3)

    # The default control noise limits: we will add noise to the final torques. the default_rfi_lim defines
    # the default limit of the range of the added noise. It represented by the percentage of the control limits.
    # noise = uniform(-rfi_lim * torque_limits, rfi_lim * torque_limits)
    default_rfi_lim = 0.1

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # domain randomization config
    # events: OmniH2OEventCfg

    # Recovery Counter for Pushed robot: Give steps for the robot to stabilize
    recovery_count = 60

    # Termination conditions
    gravity_x_threshold = 0.7
    gravity_y_threshold = 0.7
    max_ref_motion_dist = 1.5

    # reward scales
    # rewards: RewardCfg = RewardCfg()

    # motion and skeleton files
    reference_motion_path = os.path.join(TEST_DATA_DIR, "stable_punch.pkl")
    skeleton_path = os.path.join(TEST_DATA_DIR, "h1.xml")

    # When we resample reference motions
    resample_motions = True  # if we want to resample reference motions
    resample_motions_for_envs_interval_s = 1000  # How many seconds between we resample the reference motions

    # observation noise
    add_policy_obs_noise = True
    policy_obs_noise_level = 1.0
    policy_obs_noise_scales = {
        "body_pos": 0.01,  # body pos in cartesian space: 19x3
        "body_rot": 0.01,  # body pos in cartesian space: 19x3
        "body_lin_vel": 0.01,  # body velocity in cartesian space: 19x3
        "body_ang_vel": 0.01,  # body velocity in cartesian space: 19x3
        "ref_body_pos_diff": 0.05,
        "ref_body_rot_diff": 0.01,
        "ref_body_pos": 0.01,
        "ref_body_rot": 0.01,
        "ref_lin_vel": 0.01,
        "ref_ang_vel": 0.01,
    }
