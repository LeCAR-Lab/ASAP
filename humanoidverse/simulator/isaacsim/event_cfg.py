
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


import isaaclab.envs.mdp as mdp

# @configclass
# class EventCfg:
#     """Configuration for events."""

#     scale_body_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#             "mass_distribution_params": (0.8, 1.2),
#             "operation": "scale",
#         },
#     )
    
#     random_joint_friction = EventTerm(
#         func=mdp.randomize_joint_parameters,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "friction_distribution_params": (0.5, 1.25),
#             "operation": "scale",
#         },
#     )

@configclass
class EventCfg:
    """Configuration for events."""
    scale_body_mass = None
    random_joint_friction = None
    

    