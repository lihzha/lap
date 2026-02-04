"""Frame transformation utilities for robot actions."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to rotation matrix."""
    rot6d = np.asarray(rot6d)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    a2_ortho = a2 - dot * b1
    b2 = a2_ortho / np.linalg.norm(a2_ortho, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


def transform_actions_to_eef_frame(
    actions: np.ndarray, initial_state: np.ndarray, dataset_name, needs_wrist_rotation: bool = False
) -> np.ndarray:
    """Transform actions from base frame to end effector frame."""
    actions = np.asarray(actions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)

    assert actions.ndim == 1
    transformed_actions = actions.copy()

    rot6d = initial_state[3:9]
    initial_rotation = rot6d_to_rotmat(rot6d)
    r_base_to_eef = initial_rotation.T

    delta_pos_base = actions[:3]
    delta_pos_eef = r_base_to_eef @ delta_pos_base

    delta_pos_eef[1] = -delta_pos_eef[1]
    delta_pos_eef[2] = -delta_pos_eef[2]

    if "jaco_play" in dataset_name:
        delta_pos_eef = np.array([delta_pos_eef[1], delta_pos_eef[0], -delta_pos_eef[2]])
    elif "berkeley_autolab_ur5" in dataset_name:
        delta_pos_eef = np.array([-delta_pos_eef[1], delta_pos_eef[0], delta_pos_eef[2]])

    transformed_actions[:3] = delta_pos_eef

    delta_rot_base = actions[3:6]
    r_delta_base = R.from_euler("xyz", delta_rot_base).as_matrix()
    r_delta_eef = r_base_to_eef @ r_delta_base @ r_base_to_eef.T
    delta_rot_eef = R.from_matrix(r_delta_eef).as_euler("xyz")
    if not needs_wrist_rotation:
        delta_rot_eef[1] = -delta_rot_eef[1]
        delta_rot_eef[2] = -delta_rot_eef[2]

    if (
        "furniture_bench_dataset_converted_externally_to_rlds" in dataset_name
        or "austin" in dataset_name
        or "fmb" in dataset_name
        or "viola" in dataset_name
    ):
        delta_rot_eef[1] = -delta_rot_eef[1]
        delta_rot_eef[2] = -delta_rot_eef[2]
    elif "berkeley_autolab_ur5" in dataset_name:
        delta_rot_eef[1] = -delta_rot_eef[1]

    transformed_actions[3:6] = delta_rot_eef
    return transformed_actions


def transform_actions_from_eef_frame(
    actions: np.ndarray, initial_state: np.ndarray, dataset_name: str = ""
) -> np.ndarray:
    """Transform actions from end effector frame to base frame."""
    actions = np.asarray(actions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)
    if len(initial_state.shape) == 2:
        assert initial_state.shape[0] == 1
        initial_state = initial_state[0]

    if actions.ndim == 1:
        actions = actions[None, :]

    transformed_actions = actions.copy()

    if len(initial_state) == 7:
        euler = initial_state[3:6]
        initial_rotation = R.from_euler("xyz", euler).as_matrix()
    else:
        rot6d = initial_state[3:9]
        initial_rotation = rot6d_to_rotmat(rot6d)

    r_eef_to_base = initial_rotation

    for i in range(len(transformed_actions)):
        delta_pos_eef = actions[i, :3].copy()

        if "jaco_play" in dataset_name:
            delta_pos_eef = np.array([delta_pos_eef[1], delta_pos_eef[0], -delta_pos_eef[2]])
        elif "berkeley_autolab" in dataset_name:
            delta_pos_eef = np.array([delta_pos_eef[1], -delta_pos_eef[0], delta_pos_eef[2]])
        else:
            delta_pos_eef[1] = -delta_pos_eef[1]
            delta_pos_eef[2] = -delta_pos_eef[2]

        delta_pos_base = r_eef_to_base @ delta_pos_eef
        transformed_actions[i, :3] = delta_pos_base

        if actions.shape[-1] >= 6:
            delta_rot_eef = actions[i, 3:6].copy()

            if "furniture_bench" in dataset_name or "utaustin" in dataset_name or "fmb" in dataset_name:
                delta_rot_eef[1] = -delta_rot_eef[1]
                delta_rot_eef[2] = -delta_rot_eef[2]
            elif "berkeley_autolab" in dataset_name:
                delta_rot_eef[1] = -delta_rot_eef[1]
            elif "jaco_play" in dataset_name:
                pass
            else:
                delta_rot_eef[1] = -delta_rot_eef[1]
                delta_rot_eef[2] = -delta_rot_eef[2]

            r_delta_eef = R.from_euler("xyz", delta_rot_eef).as_matrix()
            r_delta_base = r_eef_to_base @ r_delta_eef @ r_eef_to_base.T
            delta_rot_base = R.from_matrix(r_delta_base).as_euler("xyz")
            transformed_actions[i, 3:6] = delta_rot_base

    return transformed_actions

