from enum import Enum
from enum import IntEnum

import tensorflow as tf

# Import rotation utilities directly to avoid package-level import side effects
import lap.datasets.utils.rotation_utils as rotation_utils


# Note: Both DROID and OXE use roll-pitch-yaw convention (extrinsic XYZ).
# Note: quaternion is in xyzw order.
# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    NONE = -1  # No Proprioceptive State
    POS_EULER = 1  # EEF XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1)  Note: no <PAD>
    POS_QUAT = 2  # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3  # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4  # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    EEF_R6 = 5  # EEF XYZ (3) + R6 (6) + Gripper Open/Close (1)


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    EEF_POS = 1  # EEF Delta XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1).
    JOINT_POS = 2  # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4  # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    ABS_EEF_POS = 5  # EEF Absolute XYZ (3) + Roll-Pitch-Yaw extrinsic XYZ (3) + Gripper Open/Close (1)


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


def state_encoding_to_type(encoding: StateEncoding) -> str:
    """Map StateEncoding to human-readable state type string.

    Args:
        encoding: The StateEncoding enum value

    Returns:
        State type string: "none", "joint_pos", or "eef_pose"
    """
    if encoding == StateEncoding.NONE:
        return "none"
    if encoding in (StateEncoding.JOINT, StateEncoding.JOINT_BIMANUAL):
        return "joint_pos"
    if encoding in (StateEncoding.POS_EULER, StateEncoding.POS_QUAT, StateEncoding.EEF_R6):
        return "eef_pose"
    raise ValueError(f"Unknown StateEncoding: {encoding}")


# euler_xyz_to_rot is imported from rotation_utils as euler_xyz_to_rot_np


def extract_episode_path_from_file_path(file_path):
    """Extract episode path from a full file path using regex.

    Removes everything up to and including 'r2d2-data/' or
    'r2d2-data-full/', then trims anything from '/trajectory' onwards.
    """
    # Strip dataset prefix up to r2d2-data or r2d2-data-full
    rel = tf.strings.regex_replace(
        file_path,
        r"^.*r2d2-data(?:-full)?/",
        "",
    )
    # Remove trailing '/trajectory...' suffix
    episode_path = tf.strings.regex_replace(
        rel,
        r"/trajectory.*$",
        "",
    )
    return episode_path


def project_in_bounds(xyz, intr4, extr44):
    xyz = tf.cast(xyz, tf.float32)
    intr4 = tf.cast(intr4, tf.float32)
    extr44 = tf.cast(extr44, tf.float32)
    # xyz: [N,3], intr4: [N,4], extr44: [N,4,4]
    # Compute camera coordinates
    ones = tf.ones_like(xyz[..., :1], dtype=tf.float32)
    p_base = tf.concat([xyz, ones], axis=-1)  # [N,4]
    base_to_cam = tf.linalg.inv(extr44)
    p_cam = tf.einsum("nij,nj->ni", base_to_cam, p_base)
    z = p_cam[..., 2]
    fx = intr4[..., 0]
    fy = intr4[..., 1]
    cx = intr4[..., 2]
    cy = intr4[..., 3]
    valid = tf.logical_and(
        z > tf.constant(1e-6, tf.float32),
        tf.logical_and(fx > 0.0, fy > 0.0),
    )
    # Pixel at calibration resolution
    u = fx * (p_cam[..., 0] / z) + cx
    v = fy * (p_cam[..., 1] / z) + cy
    # Letterbox to 224x224 using same math as resize_with_pad
    Wt = tf.constant(224.0, dtype=tf.float32)
    Ht = tf.constant(224.0, dtype=tf.float32)
    Wc = tf.maximum(tf.constant(1.0, tf.float32), 2.0 * cx)
    Hc = tf.maximum(tf.constant(1.0, tf.float32), 2.0 * cy)
    ratio = tf.maximum(Wc / Wt, Hc / Ht)
    resized_w = Wc / ratio
    resized_h = Hc / ratio
    pad_w0 = (Wt - resized_w) / 2.0
    pad_h0 = (Ht - resized_h) / 2.0
    x = u * (resized_w / Wc) + pad_w0
    y = v * (resized_h / Hc) + pad_h0
    in_x = tf.logical_and(
        x >= tf.constant(0.0, tf.float32),
        x <= (Wt - tf.constant(1.0, tf.float32)),
    )
    in_y = tf.logical_and(
        y >= tf.constant(0.0, tf.float32),
        y <= (Ht - tf.constant(1.0, tf.float32)),
    )
    return tf.logical_and(valid, tf.logical_and(in_x, in_y))


def convert_state_encoding(state: tf.Tensor, from_encoding: StateEncoding, to_encoding: StateEncoding) -> tf.Tensor:
    """
    Convert state representation between different encodings.

    Args:
        state: Input state tensor
        from_encoding: Source encoding type
        to_encoding: Target encoding type

    Returns:
        Converted state tensor
    """
    if from_encoding == to_encoding:
        return state

    # Handle conversions between POS_EULER, POS_QUAT, and EEF_R6
    if from_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT} and to_encoding in {
        StateEncoding.POS_EULER,
        StateEncoding.POS_QUAT,
    }:
        return _convert_pos_euler_quat(state, from_encoding, to_encoding)
    if from_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT} and to_encoding == StateEncoding.EEF_R6:
        return _convert_pos_to_eef_r6(state, from_encoding)
    if from_encoding == StateEncoding.EEF_R6 and to_encoding in {StateEncoding.POS_EULER, StateEncoding.POS_QUAT}:
        return _convert_eef_r6_to_pos(state, to_encoding)
    # raise ValueError(f"Unsupported state encoding conversion: {from_encoding} -> {to_encoding}")
    # logging.warning(f"Unsupported state encoding conversion: {from_encoding} -> {to_encoding}")
    return state


def convert_action_encoding(
    action: tf.Tensor, from_encoding: ActionEncoding, to_encoding: ActionEncoding, to_delta_cartesian_pose: bool = False
) -> tf.Tensor:
    """
    Convert action representation between different encodings.

    Args:
        action: Input action tensor
        from_encoding: Source encoding type
        to_encoding: Target encoding type

    Returns:
        Converted action tensor
    """
    if from_encoding == to_encoding:
        return action

    # Handle conversions between EEF_POS and EEF_R6
    if (from_encoding in (ActionEncoding.ABS_EEF_POS, ActionEncoding.EEF_POS)) and to_encoding == ActionEncoding.EEF_R6:
        return _convert_eef_pos_to_eef_r6(action)
    if from_encoding == ActionEncoding.EEF_R6 and to_encoding in (ActionEncoding.ABS_EEF_POS, ActionEncoding.EEF_POS):
        return _convert_eef_r6_to_eef_pos(action)
    if from_encoding == ActionEncoding.ABS_EEF_POS and to_encoding == ActionEncoding.EEF_POS:
        return action
    if from_encoding == ActionEncoding.EEF_POS and to_encoding == ActionEncoding.ABS_EEF_POS:
        return action
    raise ValueError(f"Unsupported action encoding conversion: {from_encoding} -> {to_encoding}")


def _convert_pos_euler_quat(state: tf.Tensor, from_encoding: StateEncoding, to_encoding: StateEncoding) -> tf.Tensor:
    """Convert between POS_EULER and POS_QUAT encodings."""
    if from_encoding == StateEncoding.POS_EULER and to_encoding == StateEncoding.POS_QUAT:
        # POS_EULER: [x, y, z, rx, ry, rz, gripper] -> POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper]
        xyz = state[..., :3]  # [..., 3]
        euler = state[..., 3:6]  # [..., 3] - rx, ry, rz
        gripper = state[..., -1:]  # [..., 1]

        # Convert euler angles to quaternion
        quat = _euler_to_quaternion(euler)

        return tf.concat([xyz, quat, gripper], axis=-1)

    if from_encoding == StateEncoding.POS_QUAT and to_encoding == StateEncoding.POS_EULER:
        # POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper] -> POS_EULER: [x, y, z, rx, ry, rz, gripper]
        xyz = state[..., :3]  # [..., 3]
        quat = state[..., 3:7]  # [..., 4] - qx, qy, qz, qw
        gripper = state[..., -1:]  # [..., 1]

        # Convert quaternion to euler angles
        euler = _quaternion_to_euler(quat)

        return tf.concat([xyz, euler, gripper], axis=-1)

    raise ValueError(f"Unsupported conversion: {from_encoding} -> {to_encoding}")


def _convert_pos_to_eef_r6(state: tf.Tensor, from_encoding: StateEncoding) -> tf.Tensor:
    """Convert POS_EULER or POS_QUAT to EEF_R6 encoding."""
    dtype = state.dtype
    if from_encoding == StateEncoding.POS_EULER:
        # POS_EULER: [x, y, z, rx, ry, rz, gripper] -> EEF_R6: [x, y, z, r11, r12, r13, r21, r22, r23, gripper]
        xyz = state[..., :3]  # [..., 3]
        euler = state[..., 3:6]  # [..., 3] - rx, ry, rz
        gripper = state[..., -1:]  # [..., 1]

        # Convert euler angles to rotation matrix (first 6 elements of 3x3 matrix)
        rot_matrix = _euler_to_rotation_matrix(euler)  # [..., 3, 3]
        r6 = _rotation_matrix_to_r6(rot_matrix)

        return tf.concat([tf.cast(xyz, dtype), tf.cast(r6, dtype), tf.cast(gripper, dtype)], axis=-1)

    if from_encoding == StateEncoding.POS_QUAT:
        # POS_QUAT: [x, y, z, qx, qy, qz, qw, gripper] -> EEF_R6: [x, y, z, r11, r12, r13, r21, r22, r23, gripper]
        xyz = state[..., :3]  # [..., 3]
        quat = state[..., 3:7]  # [..., 4] - qx, qy, qz, qw
        gripper = state[..., -1:]  # [..., 1]

        # Convert quaternion to rotation matrix (first 6 elements of 3x3 matrix)
        rot_matrix = _quaternion_to_rotation_matrix(quat)  # [..., 3, 3]
        r6 = _rotation_matrix_to_r6(rot_matrix)

        return tf.concat([tf.cast(xyz, dtype), tf.cast(r6, dtype), tf.cast(gripper, dtype)], axis=-1)

    raise ValueError(f"Unsupported conversion from {from_encoding} to EEF_R6")


def _convert_eef_r6_to_eef_pos(action: tf.Tensor) -> tf.Tensor:
    """Convert EEF_R6 action to EEF_POS action."""
    # EEF_R6: [dx, dy, dz, dr11, dr12, dr13, dr21, dr22, dr23, gripper] -> EEF_POS: [dx, dy, dz, drx, dry, drz, gripper]
    xyz_delta = action[..., :3]  # [..., 3]
    r6_delta = action[..., 3:9]  # [..., 6] - dr11, dr12, dr13, dr21, dr22, dr23
    gripper = action[..., -1:]  # [..., 1]

    rot_matrix = _r6_to_rotation_matrix(r6_delta)

    # Convert rotation matrix to euler angles
    euler_delta = _rotation_matrix_to_euler(rot_matrix)

    return tf.concat([xyz_delta, euler_delta, gripper], axis=-1)


def _convert_eef_r6_to_pos(state: tf.Tensor, to_encoding: StateEncoding) -> tf.Tensor:
    """Convert EEF_R6 encoding to POS_EULER or POS_QUAT."""
    xyz = state[..., :3]  # [..., 3]
    r6 = state[..., 3:9]  # [..., 6] - r11, r12, r13, r21, r22, r23
    gripper = state[..., -1:]  # [..., 1]

    rot_matrix = _r6_to_rotation_matrix(r6)

    if to_encoding == StateEncoding.POS_EULER:
        # Convert rotation matrix to euler angles
        euler = _rotation_matrix_to_euler(rot_matrix)
        return tf.concat([xyz, euler, gripper], axis=-1)

    if to_encoding == StateEncoding.POS_QUAT:
        # Convert rotation matrix to quaternion
        quat = _rotation_matrix_to_quaternion(rot_matrix)
        return tf.concat([xyz, quat, gripper], axis=-1)

    raise ValueError(f"Unsupported conversion from EEF_R6 to {to_encoding}")


def _convert_eef_pos_to_eef_r6(action: tf.Tensor) -> tf.Tensor:
    """Convert EEF_POS action to EEF_R6 action."""
    # EEF_POS: [dx, dy, dz, drx, dry, drz, gripper] -> EEF_R6: [dx, dy, dz, dr11, dr12, dr13, dr21, dr22, dr23, gripper]
    xyz_delta = action[..., :3]  # [..., 3]
    euler_delta = action[..., 3:6]  # [..., 3] - drx, dry, drz
    gripper = action[..., -1:]  # [..., 1]

    dtype = action.dtype

    # Convert euler angle deltas to rotation matrix deltas
    rot_delta = _euler_to_rotation_matrix(euler_delta)  # [..., 3, 3]
    r6_delta = _rotation_matrix_to_r6(rot_delta)

    return tf.concat([tf.cast(xyz_delta, dtype), tf.cast(r6_delta, dtype), tf.cast(gripper, dtype)], axis=-1)


# =============================================================================
# Rotation/Quaternion Helper Functions
# =============================================================================
# NOTE: The core implementations have been moved to rotation_utils.py
# The following are thin wrappers


def euler_xyz_to_rot(rx, ry, rz):
    """Build rotation matrix from XYZ extrinsic rotations (NumPy version)."""
    return rotation_utils.euler_xyz_to_rot_np(rx, ry, rz)


def _euler_to_quaternion(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles (rx, ry, rz) to quaternion (qx, qy, qz, qw)."""
    return rotation_utils.euler_to_quaternion(euler)


def _euler_to_rotation_matrix(euler: tf.Tensor) -> tf.Tensor:
    """Convert euler angles (rx, ry, rz) to rotation matrix."""
    return rotation_utils.euler_to_rotation_matrix(euler)


def _quaternion_to_euler(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (qx, qy, qz, qw) to euler angles."""
    return rotation_utils.quaternion_to_euler(quat)


def _quaternion_to_rotation_matrix(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (qx, qy, qz, qw) to rotation matrix."""
    return rotation_utils.quaternion_to_rotation_matrix(quat)


def _rotation_matrix_to_euler(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotation matrix to Euler angles."""
    return rotation_utils.rotation_matrix_to_euler(rot_matrix)


def _rotation_matrix_to_quaternion(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotation matrix to quaternion."""
    return rotation_utils.rotation_matrix_to_quaternion(rot_matrix)


def _rotation_matrix_to_r6(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Flatten the first two rotation matrix rows into the 6D (R6) representation."""
    return rotation_utils.rotation_matrix_to_r6(rot_matrix)


def _r6_to_rotation_matrix(r6: tf.Tensor) -> tf.Tensor:
    """Reconstruct an orthonormal rotation matrix from the 6D (R6) representation."""
    return rotation_utils.r6_to_rotation_matrix(r6)
