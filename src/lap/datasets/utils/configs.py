"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

"""

from lap.datasets.utils.helpers import ActionEncoding
from lap.datasets.utils.helpers import StateEncoding

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "fractal20220817_data": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_obs_keys": ["eef_state", "gripper_closed"],
        # "state_encoding": StateEncoding.POS_QUAT,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "clip_function_input/base_pose_tool_reached",
            # "gripper_closed",
            "state"
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_v2_oxe": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "secondary": None,
            "wrist": "rgb_gripper",
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper",
        },
        "state_obs_keys": ["state_eef", "state_gripper"],  # done
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "present/xyz",
            # "present/axis_angle",
            "state",
            # "present/sensed_close",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },  # done
    "berkeley_gnm_recon": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
            "wrist": "wrist_image_left",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "cartesian_position",
            "gripper_position",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "fmb": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_side_2",
            "wrist": "image_wrist_2",
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_side_2_depth",
            "wrist": "image_wrist_2_depth",
        },
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dobbe": {
        "image_obs_keys": {"primary": "wrist_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_object_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_combined": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "sample_r1_lite": {
        "image_obs_keys": {
            "primary": "image_camera_head",
            "secondary": None,
            "wrist": "image_camera_wrist_left",
            "wrist_right": "image_camera_wrist_right",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "joint_position_torso",
            "joint_position_arm_left",
            "joint_position_arm_right",
            "gripper_state_left",
            "gripper_state_right",
        ],
        # "state_encoding": StateEncoding.JOINT_BIMANUAL,
        # "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "agibot_large_dataset": {
        "image_obs_keys": {
            "primary": "head_image",
            "secondary": None,
            "wrist": "image_camera_wrist_left",
            "wrist_right": "image_camera_wrist_right",
        },
        "depth_obs_keys": {
            "primary": None,
            "secondary": None,
            "wrist": "hand_left_image",
            "wrist_right": "hand_right_image",
        },
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.JOINT_BIMANUAL,
        # "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "molmoact_dataset": {
        "image_obs_keys": {
            "primary": "first_view_image",
            "secondary": "second_view_image",
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}


OXE_DATASET_METADATA = {
    "fractal20220817_data": {
        "control_frequency": 3,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "kuka": {
        "control_frequency": 10,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "bridge_v2_oxe": {
        "control_frequency": 5,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "taco_play": {
        "control_frequency": 15,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "jaco_play": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "roboturk": {
        "control_frequency": 5,  # 10
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "viola": {
        "control_frequency": 20,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_autolab_ur5": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "bc_z": {
        "control_frequency": 30,  # actually 10, but robot moves too slow
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "Yes",
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "utaustin_mutex": {
        "control_frequency": 20,
        "language_annotations": "Natural Language annotations generate with GPT4 and followed by human correction.",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_fanuc_manipulation": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "cmu_stretch": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "berkeley_gnm_recon": {
        "control_frequency": 3,
        "language_annotations": "Natural",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "Yes",
    },
    "berkeley_gnm_cory_hall": {
        "control_frequency": 5,
        "language_annotations": "Natural",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "No",
    },
    "berkeley_gnm_sac_son": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "No",
    },
    "droid": {
        "control_frequency": 15,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "dobbe": {
        "control_frequency": 15,  # 3.75
        "language_annotations": "Natural",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "fmb": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "plex_robosuite": {
        "control_frequency": 20,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "sample_r1_lite": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "agibot_large_dataset": {
        "control_frequency": 30,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Bi-Manual",
        "has_suboptimal": "No",
    },
    "molmoact_dataset": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_10_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_spatial_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_object_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_goal_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
}

for dataset_name, metadata in OXE_DATASET_METADATA.items():
    if dataset_name in OXE_DATASET_CONFIGS:
        OXE_DATASET_CONFIGS[dataset_name].update(metadata)
