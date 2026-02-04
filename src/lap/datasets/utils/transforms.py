"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any

import tensorflow as tf

# Import constants
# Import helper functions from transform_helpers module
from lap.datasets.utils.transform_helpers import binarize_gripper_actions
from lap.datasets.utils.transform_helpers import build_matrix_state_transform
from lap.datasets.utils.transform_helpers import compute_padded_movement_actions
from lap.datasets.utils.transform_helpers import fill_empty_language_instruction
from lap.datasets.utils.transform_helpers import invert_gripper_actions
from lap.datasets.utils.transform_helpers import rel2abs_gripper_actions
from lap.datasets.utils.transform_helpers import rescale_action_with_bound


def normalize_action(value: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    """Normalize values using mean and standard deviation."""
    return (value - mean) / std


def wxyz_to_r6(quaternion: tf.Tensor) -> tf.Tensor:
    """
    Converts quaternion (wxyz format) to r6 representation (first 6 elements of rotation matrix).

    The r6 representation contains the first two rows of the 3x3 rotation matrix:
    [r11, r12, r13, r21, r22, r23]

    Args:
        quaternion: Quaternion tensor of shape (..., 4) in [w, x, y, z] format

    Returns:
        R6 representation of shape (..., 6) containing [r11, r12, r13, r21, r22, r23]
    """
    import tensorflow_graphics.geometry.transformation as tft

    # tensorflow_graphics expects quaternions in [x, y, z, w] format
    # Convert from [w, x, y, z] to [x, y, z, w]
    quat_xyzw = tf.concat([quaternion[..., 1:4], quaternion[..., 0:1]], axis=-1)

    # Convert to rotation matrix (shape: ..., 3, 3)
    rotation_matrix = tft.rotation_matrix_3d.from_quaternion(quat_xyzw)

    # Extract first two rows (6 elements): r11, r12, r13, r21, r22, r23
    r6 = tf.concat([rotation_matrix[..., 0, :], rotation_matrix[..., 1, :]], axis=-1)

    return r6


def axis_angle_to_r6(axis_angle: tf.Tensor) -> tf.Tensor:
    """
    Converts axis-angle representation to r6 representation (first 6 elements of rotation matrix).

    The r6 representation contains the first two rows of the 3x3 rotation matrix:
    [r11, r12, r13, r21, r22, r23]

    Args:
        axis_angle: Axis-angle tensor of shape (..., 3) where the magnitude is the angle
                    and the direction is the axis

    Returns:
        R6 representation of shape (..., 6) containing [r11, r12, r13, r21, r22, r23]
    """
    import tensorflow_graphics.geometry.transformation as tft

    # Compute angle as the norm of the axis_angle vector (shape: ..., 1)
    angle = tf.norm(axis_angle, axis=-1, keepdims=True)

    # Compute axis as normalized axis_angle (handle zero angle case)
    # When angle is zero, any axis works, so we use [1, 0, 0]
    safe_angle = tf.where(angle < 1e-8, tf.ones_like(angle), angle)
    axis = axis_angle / safe_angle

    # For zero angles, use identity rotation (set axis to [1, 0, 0])
    axis = tf.where(angle < 1e-8, tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype), axis)

    # Convert to rotation matrix (shape: ..., 3, 3)
    # angle should have shape (..., 1) as required by the API
    rotation_matrix = tft.rotation_matrix_3d.from_axis_angle(axis, angle)

    # Extract first two rows (6 elements): r11, r12, r13, r21, r22, r23
    r6 = tf.concat([rotation_matrix[..., 0, :], rotation_matrix[..., 1, :]], axis=-1)

    return r6


def axis_angle_to_extrinsic_xyz_euler(axis_angle: tf.Tensor) -> tf.Tensor:
    """
    Converts axis-angle representation to extrinsic XYZ Euler angles.

    Args:
        axis_angle: Axis-angle tensor of shape (..., 3) where the magnitude is the angle
                    and the direction is the axis

    Returns:
        Euler angles of shape (..., 3) in extrinsic XYZ order.
    """
    import tensorflow_graphics.geometry.transformation as tft

    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = tf.where(angle < 1e-8, tf.ones_like(angle), angle)
    axis = axis_angle / safe_angle
    axis = tf.where(angle < 1e-8, tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype), axis)

    rotation_matrix = tft.rotation_matrix_3d.from_axis_angle(axis, angle)

    r20 = rotation_matrix[..., 2, 0]
    r21 = rotation_matrix[..., 2, 1]
    r22 = rotation_matrix[..., 2, 2]
    r10 = rotation_matrix[..., 1, 0]
    r00 = rotation_matrix[..., 0, 0]

    pitch = tf.asin(tf.clip_by_value(-r20, -1.0, 1.0))
    roll = tf.atan2(r21, r22)
    yaw = tf.atan2(r10, r00)

    return tf.stack([roll, pitch, yaw], axis=-1)


# === Bridge-V2 =>> Dataset-Specific Transform ===
def rescale_bridge_action(action: tf.Tensor) -> tf.Tensor:
    """
    Rescales Bridge action to span RT-1 action space.

    Values taken from
    https://github.com/Asap7772/rt1_eval/blob/2fad77e9bf4def2ef82604d445270f83475e9726/kitchen_eval/rt1_wrapper.py#L39
    """
    # Rescale world_vector (xyz position delta)
    world_vector = rescale_action_with_bound(
        action[:, :3],
        low=-0.05,
        high=0.05,
        safety_margin=0.01,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )
    # Rescale rotation_delta
    rotation_delta = rescale_action_with_bound(
        action[:, 3:6],
        low=-0.25,
        high=0.25,
        safety_margin=0.01,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )
    return tf.concat([world_vector, rotation_delta, action[:, 6:]], axis=-1)


def relabel_bridge_actions(traj: dict[str, Any]) -> dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


def bridge_v2_oxe_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    # for key in trajectory:
    #     if key == "traj_metadata":
    #         continue
    #     if key in ["observation", "action"]:
    #         for key2 in trajectory[key]:
    #             trajectory[key][key2] = trajectory[key][key2][1:]
    #     else:
    #         trajectory[key] = trajectory[key][1:]

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"]["world_vector"],
    #         trajectory["action"]["rotation_delta"],
    #         tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
    #     ),
    #     axis=-1,
    # )
    # # print(trajectory.keys(), trajectory['observation'].keys())
    # trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    # trajectory = relabel_bridge_actions(trajectory)
    # trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    # trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    # print("bridge", trajectory.keys())
    # return trajectory
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory:
        if key == "traj_metadata":
            continue
        if key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )

    # trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = tf.clip_by_value(trajectory["observation"]["state"][:, -1:], 0, 1)

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["EEF_state"])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["EEF_state"],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def bridge_orig_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory:
        if key == "traj_metadata":
            continue
        if key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )

    # print(trajectory.keys(), trajectory['observation'].keys())
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def ppgm_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["gripper_closed"] = tf.clip_by_value(
        invert_gripper_actions(trajectory["observation"]["gripper_closed"]), 0, 1
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["base_pose_tool_reached"][:, 3:7]),
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["eef_state"])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["eef_state"],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def kuka_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1,))
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # gripper_abs_state = rel2abs_gripper_actions(trajectory["observation"]["gripper_closed"])
    gripper_abs_state = invert_gripper_actions(trajectory["observation"]["gripper_closed"])

    # Create EEF state with xyz + euler angles + gripper
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["clip_function_input/base_pose_tool_reached"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["clip_function_input/base_pose_tool_reached"][:, 3:7]),
            tf.clip_by_value(tf.reshape(gripper_abs_state, (-1, 1)), 0, 1),
        ),
        axis=-1,
    )

    gripper_action_abs = rel2abs_gripper_actions(trajectory["action"]["gripper_closedness_action"][:, 0])

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, gripper_action_abs[:, None]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            gripper_action_abs[:, None],
        ],
        axis=1,
    )

    return trajectory


def rescale_taco_play_action(action: tf.Tensor) -> tf.Tensor:
    """
    Rescales Taco Play actions based on measured per dimension ranges.

    Args:
        action: Action tensor with shape (..., 7) containing [xyz, rpy, gripper]
    """
    # Rotation Delta (indices 3:6)
    rd_lows = tf.constant([-3.2, -0.8, -1.8])
    rd_highs = tf.constant([3.2, 0.2, 2.5])
    rotation_delta = (action[:, 3:6] - rd_lows) / (rd_highs - rd_lows) * 2 - 1
    rotation_delta = tf.clip_by_value(rotation_delta, -1 + 0.01, 1 - 0.01)

    # World Vector (indices 0:3)
    wv_lows = tf.constant([0.0, -0.5, 0.0])
    wv_highs = tf.constant([0.8, 0.7, 0.6])
    world_vector = (action[:, :3] - wv_lows) / (wv_highs - wv_lows) * 2 - 1
    world_vector = tf.clip_by_value(world_vector, -1 + 0.01, 1 - 0.01)

    return tf.concat([world_vector, rotation_delta, action[:, 6:]], axis=-1)


def taco_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    # trajectory["observation"]["state_gripper"] = (
    #     (trajectory["observation"]["robot_obs"][:, 6:7] + 1) / 2 - 0.459656
    # ) / (0.5 - 0.459656)

    trajectory["observation"]["state_gripper"] = tf.clip_by_value(
        12.3903 * trajectory["observation"]["robot_obs"][:, 6:7], 0, 1
    )
    # trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][:, 7:8]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value((trajectory["action"][:, -1:] + 1) / 2, 0, 1),
        ),
        axis=-1,
    )

    # # Apply action rescaling from RT-1 preprocessing
    # trajectory["action"] = rescale_taco_play_action(trajectory["action"])

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state_eef"])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state_eef"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def jaco_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from lap.datasets.utils.rotation_utils import coordinate_transform_jaco

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["observation"]["state_eef"] = coordinate_transform_jaco(
        trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    )

    trajectory["observation"]["state_gripper"] = tf.clip_by_value(
        trajectory["observation"]["end_effector_cartesian_pos"][:, -1:] * 4.33, 0, 1
    )

    # trajectory["observation"]["state_gripper"] = gripper_action[:, None]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state_eef"])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state_eef"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def berkeley_cable_routing_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"]["world_vector"],
    #         trajectory["action"]["rotation_delta"],
    #         gripper_action,
    #     ),
    #     axis=-1,
    # )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy

    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(trajectory["observation"]["ee_states"][:, -16:], [-1, 4, 4])
    # state_matrix = tf.transpose(state_matrix, [0, 2, 1])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])  # Transpose to convert column-major to row-major
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), tf.clip_by_value(trajectory["observation"]["gripper_states"] / 0.079, 0, 1)),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, gripper_action], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            gripper_action,
        ],
        axis=1,
    )

    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["state"][:, 3:7]),
            tf.clip_by_value(invert_gripper_actions(trajectory["observation"]["state"][:, -1:]), 0, 1),
        ),
        axis=-1,
    )

    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def toto_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = invert_gripper_actions(trajectory["observation"]["state"][:, -3:-2])
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Transform for austin_buds dataset with matrix state and empty language handling."""
    return build_matrix_state_transform(
        trajectory,
        state_matrix_key="state",
        matrix_slice=slice(-16, None),
        gripper_key="state",
        gripper_slice=slice(7, 8),
        gripper_scale=0.079,
        invert_gripper_action=True,
        clip_gripper_action=True,
        fill_empty_lang=True,
        use_deterministic_fallback=True,
    )


def nyu_franka_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    print("nyu", trajectory.keys())
    print("nyu obs", trajectory["observation"].keys())

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def droid_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    cartesian = tf.cast(trajectory["observation"]["cartesian_position"], tf.float32)
    gripper = trajectory["observation"]["gripper_position"]

    gripper = tf.cond(
        tf.equal(tf.rank(cartesian), tf.rank(gripper)),
        lambda: gripper,  # same rank → no change
        lambda: tf.expand_dims(gripper, axis=-1),  # add new axis if rank differs
    )

    trajectory["state"] = tf.concat(
        [cartesian, binarize_gripper_actions(invert_gripper_actions(gripper), threshold=0.5)], axis=-1
    )

    gripper_actions = binarize_gripper_actions(
        invert_gripper_actions(trajectory["action_dict"]["gripper_position"]), threshold=0.5
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(cartesian)

    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, tf.clip_by_value(gripper_actions[:, -1:], 0, 1)],
        axis=1,
    )
    trajectory["action"] = tf.concat(
        [
            cartesian,
            tf.clip_by_value(gripper_actions[:, -1:], 0, 1),
        ],
        axis=1,
    )

    return trajectory


def maniskill_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["state"][:, 3:7]),
            tf.clip_by_value(trajectory["observation"]["state"][:, -1:] / 0.079, 0, 1),
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))],
        axis=1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ],
        axis=1,
    )

    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # from lap.datasets.oxe_utils.data_utils import euler_diff

    # trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]

    # movement_actions = tf.concat(
    #     (
    #         trajectory["action"][1:, :3] - trajectory["action"][:-1, :3],
    #         euler_diff(
    #             trajectory["action"][1:, 3:6],
    #             trajectory["action"][:-1, 3:6],
    #         ),
    #         trajectory["action"][1:, 6:7],
    #     ),
    #     axis=-1,
    # )
    # traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    # traj_truncated["action"] = movement_actions
    # return traj_truncated
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Transform for austin_sailor dataset with matrix state and empty language handling.

    Note: Uses state_ee for matrix extraction but state for gripper.
    """
    from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy

    # Extract state from transformation matrix (state_ee key)
    state_matrix = tf.reshape(trajectory["observation"]["state_ee"][:, -16:], [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), tf.clip_by_value(trajectory["observation"]["state"][:, -1:] / 0.079, 0, 1)),
        axis=-1,
    )

    # Process gripper action
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))

    # Compute movement actions
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, gripper_action], axis=1)
    trajectory["action"] = tf.concat([trajectory["observation"]["state"][:, :6], gripper_action], axis=1)

    # Fill empty language instructions
    return fill_empty_language_instruction(trajectory, use_deterministic_fallback=True)


def austin_sirius_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Transform for austin_sirius dataset with matrix state and empty language handling.

    Note: Uses random fallback (not deterministic) for empty language instructions.
    """
    from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy

    # Extract state from transformation matrix (state_ee key)
    state_matrix = tf.reshape(trajectory["observation"]["state_ee"][:, -16:], [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), tf.clip_by_value(trajectory["observation"]["state"][:, -1:] / 0.079, 0, 1)),
        axis=-1,
    )

    # Process gripper action
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))

    # Compute movement actions
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, gripper_action], axis=1)
    trajectory["action"] = tf.concat([trajectory["observation"]["state"][:, :6], gripper_action], axis=1)

    # Fill empty language instructions with random fallback
    return fill_empty_language_instruction(trajectory, use_deterministic_fallback=False)


def bc_z_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from lap.datasets.utils.rotation_utils import coordinate_transform_bcz

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"], tf.float32))[:, :1],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    present_euler = axis_angle_to_extrinsic_xyz_euler(trajectory["observation"]["present/axis_angle"][:, :3])
    trajectory["observation"]["state"] = tf.concat(
        (
            coordinate_transform_bcz(
                tf.concat(
                    (
                        trajectory["observation"]["present/xyz"][:, :3],
                        present_euler,
                    ),
                    axis=-1,
                ),
            ),
            tf.clip_by_value(
                invert_gripper_actions(trajectory["observation"]["present/sensed_close"])[:, :1] / 0.8, 0, 1
            ),  # max seems to be 0.8
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from lap.datasets.utils.rotation_utils import zxy_to_xyz

    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            zxy_to_xyz(trajectory["action"][:, 3:6]),
            tf.clip_by_value(invert_gripper_actions(trajectory["action"][:, -1:]), 0, 1),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            zxy_to_xyz(trajectory["observation"]["state"][:, 3:6]),
            invert_gripper_actions(trajectory["observation"]["state"][:, -1:]),
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def asu_table_top_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy

    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(trajectory["observation"]["state"][:, -16:], [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])  # Transpose to convert column-major to row-major
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), tf.clip_by_value(trajectory["observation"]["state"][:, 7:8] / 0.079, 0, 1)),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def molmoact_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["language_action"] = tf.concat(
        (
            trajectory["action"][:, :-1],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :-1],
            invert_gripper_actions(trajectory["observation"]["state"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=1,
    )
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    # trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["language_action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    # return trajectory

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_state"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["end_effector_state"][:, 3:7]),
            tf.clip_by_value(invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]), 0, 1),
        ),
        axis=-1,
    )

    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ],
        axis=1,
    )

    return trajectory

    # movement_actions = tf.concat(
    #     (
    #         trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
    #         euler_diff(
    #             trajectory["observation"]["state"][1:, 3:6],
    #             trajectory["observation"]["state"][:-1, 3:6],
    #         ),
    #     ),
    #     axis=-1,
    # )
    # traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    # traj_truncated["action"] = tf.concat([movement_actions, trajectory["observation"]["state"][:-1, -1:]], axis=1)
    # return traj_truncated


def cmu_playing_with_food_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = (trajectory["observation"]["state"][:, -1:] + 3.14) / 6.28
    # trajectory["action"] = (trajectory["action"][..., :-1] + 3.14) / 6.28
    # gripper_action = trajectory["action"][..., :-2:-1]

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["eef_state"])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, trajectory["observation"]["gripper_state"]], axis=1
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["eef_state"][:, :6],
            trajectory["observation"]["gripper_state"],
        ],
        axis=1,
    )

    return trajectory


def gnm_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
            tf.zeros_like(trajectory["observation"]["state"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["state"] = tf.cast(trajectory["observation"]["state"], tf.float32)
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)

    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, tf.zeros_like(trajectory["action"][:, :1])], axis=1
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ],
        axis=1,
    )

    return trajectory


def fmb_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["eef_pose"][:, 3:7]),
            tf.clip_by_value(invert_gripper_actions(trajectory["observation"]["state_gripper_pose"][..., None]), 0, 1),
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["proprio"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, invert_gripper_actions(trajectory["action"][:, -1:])], axis=1
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["proprio"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=1,
    )

    return trajectory


def dobbe_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension

    from lap.datasets.utils.rotation_utils import coordinate_transform_dobbe

    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            coordinate_transform_dobbe(trajectory["observation"]["state"][:, :6]),
            tf.clip_by_value(trajectory["observation"]["state"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    # Compute movement actions with zero-padding (no truncation)
    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["proprio"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, trajectory["action"][:, -1:]], axis=1)
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["proprio"][:, :6],
            trajectory["action"][:, -1:],
        ],
        axis=1,
    )

    return trajectory


def roboset_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    # Invert gripper state to match inverted gripper action convention
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :-1],
            invert_gripper_actions(tf.clip_by_value(trajectory["observation"]["state"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )

    state = trajectory["observation"]["state"]
    euler_xyz = axis_angle_to_extrinsic_xyz_euler(state[:, 3:6])
    gripper_state = tf.clip_by_value(state[:, -2:-1] / 0.04, 0, 1)
    trajectory["observation"]["state"] = tf.concat([state[:, :3], euler_xyz, gripper_state], axis=1)

    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat([padded_movement_actions, gripper_action], axis=1)
    # trajectory["action"] = tf.concat(
    #     [
    #         trajectory["action"][:, -1:],
    #         gripper_action,
    #     ],
    #     axis=1,
    # )

    return trajectory


def human_dataset_transform(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Transforms human data into the expected format by adding dummy actions.

    Args:
        sample (Dict[str, Any]): A dictionary containing human data observations.

    Returns:
        Dict[str, Any]: Transformed sample with dummy actions added.
    """
    # Extract the observation from the sample
    observation = sample["observation"]
    print("ego4d", sample.keys())
    print("ego4d obs", sample["observation"].keys())
    # print('sample["observation"]', sample["observation"]['image'].shape[0])
    # observation["state"] = tf.zeros((2, 7), dtype=tf.float32)

    # Create a dummy action tensor with all zeros
    # Assuming the action space is 7D (6D for EEF + 1D for gripper)
    # dummy_action = tf.zeros((2, 7), dtype=tf.float32)

    # Add the dummy action to the sample
    # sample["action"] = dummy_action

    # Split the observation state into EEF_state and gripper_state
    # sample["observation"]["EEF_state"] = observation["state"][:, :6]
    # sample["observation"]["gripper_state"] = observation["state"][:, -1:]

    return sample


def sample_r1_lite_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Transform for sample_r1_lite dataset with bimanual EEF pose states."""
    # Extract EEF states for both arms (left and right)
    # Assuming state contains [left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
    # Total: 3 + 3 + 1 + 3 + 3 + 1 = 14 dimensions
    left_eef_state = trajectory["observation"]["state"][:, :6]  # left xyz + rpy
    right_eef_state = trajectory["observation"]["state"][:, 7:13]  # right xyz + rpy

    # Compute movement actions with zero-padding (no truncation) for both arms
    left_movement = compute_padded_movement_actions(left_eef_state)
    right_movement = compute_padded_movement_actions(right_eef_state)

    # Combine left and right movement actions with gripper actions
    trajectory["action"] = tf.concat(
        [
            left_movement,
            trajectory["action"][:, 6:7] / 100,  # left gripper
            right_movement,
            trajectory["action"][:, 13:14] / 100,  # right gripper
        ],
        axis=1,
    )

    return trajectory


def agibot_large_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # Compute movement actions with zero-padding (no truncation) for both arms
    left_movement = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    right_movement = compute_padded_movement_actions(trajectory["observation"]["state"][:, 7:13])

    trajectory["action"] = tf.concat(
        (
            left_movement,
            invert_gripper_actions(trajectory["action"][:, 6:7]),
            right_movement,
            invert_gripper_actions(trajectory["action"][:, 13:14]),
        ),
        axis=-1,
    )

    return trajectory


def planning_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # Compute movement actions with zero-padding (no truncation) for both arms

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 3:6],
            wxyz_to_r6(trajectory["observation"]["state"][:, 6:10]),
            trajectory["observation"]["state"][:, 10:],
        ),
        axis=-1,
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, 3:6],
            axis_angle_to_r6(trajectory["action"][:, 6:9]),
            trajectory["action"][:, 9:],
        ),
        axis=-1,
    )

    return trajectory


def franka_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # Compute movement actions with zero-padding (no truncation) for both arms

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["observation"]["state"][:, 6:], 0, 1)),
        ),
        axis=-1,
    )

    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1))], axis=1
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ],
        axis=1,
    )

    return trajectory


def yam_jointpos_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # Compute movement actions with zero-padding (no truncation) for both arms

    # trajectory["observation"]["state"] = tf.concat(
    #     (
    #         trajectory["observation"]["state"][:, :6],
    #     ),
    #     axis=-1,
    # )

    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)], axis=1
    )
    # trajectory["action"] = tf.concat(
    #     [
    #         trajectory["observation"]["state"][:, :6],
    #        tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
    #     ],
    #     axis=1,
    # )

    return trajectory


def yam_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # Compute movement actions with zero-padding (no truncation) for both arms

    # trajectory["observation"]["state"] = tf.concat(
    #     (
    #         trajectory["observation"]["state"][:, :6],
    #     ),
    #     axis=-1,
    # )

    padded_movement_actions = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])
    trajectory["language_action"] = tf.concat(
        [padded_movement_actions, tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)], axis=1
    )
    trajectory["action"] = tf.concat(
        [
            trajectory["observation"]["state"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ],
        axis=1,
    )

    return trajectory


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_v2_oxe": bridge_v2_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    # "droid": droid_baseact_transform,
    "droid": droid_dataset_transform,
    "droid_100": droid_dataset_transform,
    "fmb": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    # "droid_wipe": droid_finetuning_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_10_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_half": libero_dataset_transform,
    "libero_10_no_noops_half": libero_dataset_transform,
    "libero_goal_no_noops_quad": libero_dataset_transform,
    "libero_10_no_noops_quad": libero_dataset_transform,
    "libero_combined": libero_dataset_transform,
    ### Human Dataset
    "ego4d_split_1": human_dataset_transform,
    "ego4d_split_2": human_dataset_transform,
    "ego4d_split_3": human_dataset_transform,
    "ego4d_split_4": human_dataset_transform,
    "sample_r1_lite": sample_r1_lite_dataset_transform,
    "agibot_large_dataset": agibot_large_dataset_transform,
    "molmoact_dataset": molmoact_dataset_transform,
    "planning_dataset": planning_dataset_transform,
    "franka_dataset": franka_dataset_transform,
    "yam_jointpos_dataset": yam_jointpos_dataset_transform,
    "yam_jointpos_dataset_small": yam_jointpos_dataset_transform,
    "yam_dataset": yam_dataset_transform,
    "franka_demo_dataset": franka_dataset_transform,
    "franka_towel_demo_dataset": franka_dataset_transform,
    "yam_demo_dataset": yam_dataset_transform,
    "franka_rollout_dataset": franka_dataset_transform,
    "yam_rollout_dataset": yam_dataset_transform,
}
