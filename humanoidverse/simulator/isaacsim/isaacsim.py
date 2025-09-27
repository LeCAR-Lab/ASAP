import sys
import os
from loguru import logger
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
# from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.timer import Timer

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from isaaclab_assets import H1_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.envs import ViewerCfg

import isaaclab.sim as sim_utils

from humanoidverse.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
import builtins
import inspect
import copy
from humanoidverse.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG

from humanoidverse.simulator.isaacsim.event_cfg import EventCfg

from isaaclab.managers import EventManager

from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from humanoidverse.simulator.isaacsim.events import randomize_body_com

class IsaacSim(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        self.terrain_config = config.terrain
        self.domain_rand_config = config.domain_rand
        
        sim_config: SimulationCfg = SimulationCfg(dt=1./self.simulator_config.sim.fps, 
                                           render_interval=self.simulator_config.sim.render_interval, 
                                           device=self.sim_device,
                                           physx=PhysxCfg(bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                                                          solver_type=self.simulator_config.sim.physx.solver_type,
                                                          max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                                                          max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations))
        
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
        
        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1./self.simulator_config.sim.fps}")
        logger.info(f"\tRendering step-size   : {1./self.simulator_config.sim.fps * self.simulator_config.sim.substeps}")


        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)
        
        
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=self.simulator_config.scene.num_envs, env_spacing=self.simulator_config.scene.env_spacing, replicate_physics=self.simulator_config.scene.replicate_physics)
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        
    
        
        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_config)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
        
        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.simulator_config.scene.num_envs, 3), dtype=torch.float, device="cpu")


        self.events_cfg = EventCfg()
        if self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg.scale_body_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "mass_distribution_params": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                },
            )

        # Randomize joint friction
        if self.domain_rand_config.get("randomize_friction", False):
            self.events_cfg.random_joint_friction = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": tuple(self.domain_rand_config["friction_range"]),
                    "operation": "scale",
                },
            )

        if self.domain_rand_config.get("randomize_base_com", False):
            self.events_cfg.random_base_com = EventTerm(
                func=randomize_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=[
                            "torso_link",
                        ],
                    ),
                    "distribution_params": (
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["y"][0], self.domain_rand_config["base_com_range"]["z"][0]]),
                        torch.tensor([self.domain_rand_config["base_com_range"]["x"][1], self.domain_rand_config["base_com_range"]["y"][1], self.domain_rand_config["base_com_range"]["z"][1]])
                    ),
                    "operation": "add",
                    "distribution": "uniform",
                    "num_envs": self.simulator_config.scene.num_envs,
                },
            )  

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
        
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
                
        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        
        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None


        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation


        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # print the environment information
        logger.info("Completed setting up the environment...")
        
        
    def _setup_scene(self):
        # actuators = {
        #     "legs": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
        #         effort_limit={
        #             ".*_hip_yaw": 200.0,
        #             ".*_hip_roll": 200.0,
        #             ".*_hip_pitch": 200.0,
        #             ".*_knee": 300.0,
        #             "torso": 200.0,
        #         },
        #         velocity_limit={
        #             ".*_hip_yaw": 23.0,
        #             ".*_hip_roll": 23.0,
        #             ".*_hip_pitch": 23.0,
        #             ".*_knee": 14.0,
        #             "torso": 23.0,
        #         },
        #         stiffness=0,
        #         damping=0,
        #     ),
        #     "feet": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_ankle"],
        #         effort_limit=40,
        #         velocity_limit=9.0,
        #         stiffness=0,
        #         damping=0,
        #     ),
        #     "arms": IdealPDActuatorCfg(
        #         joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
        #         effort_limit={
        #             ".*_shoulder_pitch": 40.0,
        #             ".*_shoulder_roll": 40.0,
        #             ".*_shoulder_yaw": 18.0,
        #             ".*_elbow": 18.0,
        #         },
        #         velocity_limit={
        #             ".*_shoulder_pitch": 9.0,
        #             ".*_shoulder_roll": 9.0,
        #             ".*_shoulder_yaw": 20.0,
        #             ".*_elbow": 20.0,
        #         },
        #         stiffness=0,
        #         damping=0,
        #     ),
        # }
        asset_root = self.robot_config.asset.asset_root
        asset_path = self.robot_config.asset.usd_file
        # prapare to override the spawn configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(asset_root, asset_path),
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        )
        
        # prepare to override the articulation configuration in HumanoidVerse/humanoidverse/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={
                joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()
            },
            joint_vel={".*": 0.0},
        )
       
        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")    
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping
        
        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict.keys():
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")


        # ImplicitActuatorCfg IdealPDActuatorCfg  
        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            ) for i in range(len(dof_names_list))
        }
        # import ipdb; ipdb.set_trace()
        # actuators = {
        #     dof_names_list[i]: ImplicitActuatorCfg(
        #         joint_names_expr=[dof_names_list[i]],
        #         effort_limit=dof_effort_limit_list[i],
        #         velocity_limit=dof_vel_limit_list[i],
        #         stiffness=kp_list[i],
        #         damping=kd_list[i],
        #     ) for i in range(len(dof_names_list))
        # }

        # actuators={
        # "legs": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_hip_yaw_joint": 150.0,
        #         ".*_hip_roll_joint": 150.0,
        #         ".*_hip_pitch_joint": 200.0,
        #         ".*_knee_joint": 200.0,
        #         "torso_joint": 200.0,
        #     },
        #     damping={
        #         ".*_hip_yaw_joint": 5.0,
        #         ".*_hip_roll_joint": 5.0,
        #         ".*_hip_pitch_joint": 5.0,
        #         ".*_knee_joint": 5.0,
        #         "torso_joint": 5.0,
        #     },
        # ),
        # "feet": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_ankle_joint"],
        #     effort_limit=100,
        #     velocity_limit=100.0,
        #     stiffness={".*_ankle_joint": 20.0},
        #     damping={".*_ankle_joint": 4.0},
        # ),
        # "arms": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
        #     effort_limit=300,
        #     velocity_limit=100.0,
        #     stiffness={
        #         ".*_shoulder_pitch_joint": 40.0,
        #         ".*_shoulder_roll_joint": 40.0,
        #         ".*_shoulder_yaw_joint": 40.0,
        #         ".*_elbow_joint": 40.0,
        #     },
        #     damping={
        #         ".*_shoulder_pitch_joint": 10.0,
        #         ".*_shoulder_roll_joint": 10.0,
        #         ".*_shoulder_yaw_joint": 10.0,
        #         ".*_elbow_joint": 10.0,
        #     },
        # )
        # }
        
        # import ipdb; ipdb.set_trace()
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)
        # robot_articulation_config: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)
        # robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators)
        
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
        )

        # Add a height scanner to the torso to detect the height of the terrain mesh
        height_scanner_config = RayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # Apply a grid pattern that is smaller than the resolution to only return one height value.
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        
        if (self.terrain_config.mesh_type == "heightfield") or (self.terrain_config.mesh_type == "trimesh"):
            sub_terrains = {}
            terrain_types = self.terrain_config.terrain_types
            terrain_proportions = self.terrain_config.terrain_proportions
            for terrain_type, proportion in zip(terrain_types, terrain_proportions):
                if proportion > 0:
                    if terrain_type == "flat":
                        sub_terrains[terrain_type] = terrain_gen.MeshPlaneTerrainCfg(
                            proportion=proportion
                        )
                    elif terrain_type == "rough":
                        sub_terrains[terrain_type] = terrain_gen.HfRandomUniformTerrainCfg(
                            proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
                        )
                    elif terrain_type == "low_obst":
                        sub_terrains[terrain_type] = terrain_gen.MeshRandomGridTerrainCfg(
                            proportion=proportion, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
                        )

            terrain_generator_config = TerrainGeneratorCfg(
                curriculum=self.terrain_config.curriculum,
                size=(self.terrain_config.terrain_length, self.terrain_config.terrain_width),
                border_width=self.terrain_config.border_size,
                num_rows=self.terrain_config.num_rows,
                num_cols=self.terrain_config.num_cols,
                horizontal_scale=self.terrain_config.horizontal_scale,
                vertical_scale=self.terrain_config.vertical_scale,
                slope_threshold=self.terrain_config.slope_treshold,
                use_cache=False,
                sub_terrains=sub_terrains,
            )

            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_generator_config,
                max_init_terrain_level=9,
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            # terrain_config.env_spacing = self.scene.cfg.env_spacing

        else:
            terrain_config = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.terrain_config.static_friction,
                    dynamic_friction=self.terrain_config.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
        
        self._robot = Articulation(robot_articulation_config)
        self.scene.articulations["robot"] = self._robot
        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self._height_scanner = RayCaster(height_scanner_config)
        self.scene.sensors["height_scanner"] = self._height_scanner

        
        
        self.terrain = terrain_config.class_type(terrain_config)
        self.terrain.env_origins = self.terrain.terrain_origins
        
        # import ipdb; ipdb.set_trace()

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[terrain_config.prim_path])

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))


    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            from omni.isaac.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def setup(self):
        self.sim_dt = 1. / self.simulator_config.sim.fps
        
    
    def setup_terrain(self, mesh_type):
        pass


    def load_assets(self):
        '''
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        '''

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")     
        # isaacsim only support matching joint names without "joint" postfix

        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 1.05),
        #     joint_pos={
        #         ".*_hip_yaw": 0.0,
        #         ".*_hip_roll": 0.0,
        #         ".*_hip_pitch": -0.28,  # -16 degrees
        #         ".*_knee": 0.79,  # 45 degrees
        #         ".*_ankle": -0.52,  # -30 degrees
        #         "torso": 0.0,
        #         ".*_shoulder_pitch": 0.28,
        #         ".*_shoulder_roll": 0.0,
        #         ".*_shoulder_yaw": 0.0,
        #         ".*_elbow": 0.52,
        #     },
        #     joint_vel={".*": 0.0},
        # ),

        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        #     ),
        # ),

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True) 
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)


        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, humanoidverse config)
            # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

    
        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        '''
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19], ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link', 'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        '''
        
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning("The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim.")
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"
       
        
        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names
        

    def create_envs(self, num_envs, env_origins, base_init_state):
        
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        
        return self.scene, self._robot
    
    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices
                
    def prepare_sim(self):
        self.refresh_sim_tensors() # initialize tensors

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        ############################################################################################
        # TODO: currently, we only consider the robot root state, ignore other objects's root states
        ############################################################################################
        self.all_root_states = self._robot.data.root_state_w  # (num_envs, 13)
        
        self.robot_root_states = self.all_root_states # (num_envs, 13)
        self.base_quat = self.robot_root_states[:, [4, 5, 6, 3]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids] # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        self.contact_forces = self.contact_sensor.data.net_forces_w # (num_envs, num_bodies, 3)

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        self._robot.write_root_pose_to_sim(root_states[set_env_ids, :7], set_env_ids)
        self._robot.write_root_velocity_to_sim(root_states[set_env_ids, 7:], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)
    
    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim 
        self.scene.update(dt=1./self.simulator_config.sim.fps)
    
    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller


    def render(self, sync_frame_time=True):
        pass

     # debug visualization
    def clear_lines(self):
        self.draw.clear_lines()
        self.draw.clear_points()

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        # draw a big sphere
        point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
        color_list = [(color[0], color[1], color[2], 1.0)]
        sizes = [20]
        self.draw.draw_points(point_list, color_list, sizes)

    def draw_line(self, start_point, end_point, color, env_id):
        # import ipdb; ipdb.set_trace()
        start_point_list = [(   start_point.x.item(), start_point.y.item(), start_point.z.item())]
        end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
        color_list = [(color.x, color.y, color.z, 1.0)]
        sizes = [1]
        self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)
