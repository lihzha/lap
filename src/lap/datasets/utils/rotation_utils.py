"""Rotation and quaternion utilities for RLDS datasets.

This module consolidates all rotation-related functions including:
- Euler angle conversions (extrinsic XYZ convention)
- Quaternion operations
- Rotation matrix utilities
- 6D rotation representation (R6)

Note: Both DROID and OXE use roll-pitch-yaw convention (extrinsic XYZ).
Note: Quaternions are in xyzw order unless otherwise specified.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lap.datasets.utils.constants import EPSILON


def _tf_pi(dtype=tf.float32):
    """Get pi constant with specified dtype."""
    return tf.constant(3.141592653589793, dtype=dtype)


# =============================================================================
# Basic Rotation Matrix Builders
# =============================================================================


@tf.function
def rot_x(angle: tf.Tensor) -> tf.Tensor:
    """Rotation matrix around X axis."""
    ca, sa = tf.cos(angle), tf.sin(angle)
    z = tf.zeros_like(angle)
    o = tf.ones_like(angle)
    return tf.stack(
        [
            tf.stack([o, z, z], axis=-1),
            tf.stack([z, ca, -sa], axis=-1),
            tf.stack([z, sa, ca], axis=-1),
        ],
        axis=-2,
    )


@tf.function
def rot_y(angle: tf.Tensor) -> tf.Tensor:
    """Rotation matrix around Y axis."""
    ca, sa = tf.cos(angle), tf.sin(angle)
    z = tf.zeros_like(angle)
    o = tf.ones_like(angle)
    return tf.stack(
        [
            tf.stack([ca, z, sa], axis=-1),
            tf.stack([z, o, z], axis=-1),
            tf.stack([-sa, z, ca], axis=-1),
        ],
        axis=-2,
    )


@tf.function
def rot_z(angle: tf.Tensor) -> tf.Tensor:
    """Rotation matrix around Z axis."""
    ca, sa = tf.cos(angle), tf.sin(angle)
    z = tf.zeros_like(angle)
    o = tf.ones_like(angle)
    return tf.stack(
        [
            tf.stack([ca, -sa, z], axis=-1),
            tf.stack([sa, ca, z], axis=-1),
            tf.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


# =============================================================================
# Euler <-> Rotation Matrix (Extrinsic XYZ Convention)
# =============================================================================


def euler_to_rotation_matrix(euler: tf.Tensor) -> tf.Tensor:
    """Convert extrinsic XYZ Euler angles to rotation matrix.

    Args:
        euler: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)

    Returns:
        (..., 3, 3) rotation matrix where R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    roll, pitch, yaw = tf.unstack(euler, num=3, axis=-1)

    cr, sr = tf.cos(roll), tf.sin(roll)
    cp, sp = tf.cos(pitch), tf.sin(pitch)
    cy, sy = tf.cos(yaw), tf.sin(yaw)

    # Extrinsic XYZ is equivalent to intrinsic ZYX, so R = Rz * Ry * Rx
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return tf.stack(
        [
            tf.stack([r00, r01, r02], axis=-1),
            tf.stack([r10, r11, r12], axis=-1),
            tf.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


@tf.function
def rotation_matrix_to_euler(rot_matrix: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Convert rotation matrix to extrinsic XYZ Euler angles.

    For extrinsic XYZ: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)

    Args:
        rot_matrix: (..., 3, 3) rotation matrix
        eps: Epsilon for gimbal lock handling

    Returns:
        (..., 3) tensor of Euler angles [roll, pitch, yaw]
    """
    R = tf.convert_to_tensor(rot_matrix)
    dtype = R.dtype
    eps_t = tf.cast(eps, dtype)

    r00 = R[..., 0, 0]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    sy = tf.sqrt(tf.maximum(r00 * r00 + r10 * r10, eps_t))
    singular = sy < eps_t

    roll_regular = tf.atan2(r21, r22)
    roll_singular = tf.atan2(-r12, r11)
    roll = tf.where(singular, roll_singular, roll_regular)

    pitch = tf.atan2(-r20, sy)

    yaw_regular = tf.atan2(r10, r00)
    yaw_singular = tf.zeros_like(yaw_regular)
    yaw = tf.where(singular, yaw_singular, yaw_regular)

    return tf.stack([roll, pitch, yaw], axis=-1)


# =============================================================================
# Quaternion Operations (xyzw order)
# =============================================================================


def euler_to_quaternion(euler: tf.Tensor) -> tf.Tensor:
    """Convert extrinsic XYZ Euler angles to quaternion (xyzw order).

    Args:
        euler: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)

    Returns:
        (..., 4) quaternion in xyzw order
    """
    rx, ry, rz = tf.unstack(euler, num=3, axis=-1)

    # Half angles
    cx, sx = tf.cos(rx * 0.5), tf.sin(rx * 0.5)
    cy, sy = tf.cos(ry * 0.5), tf.sin(ry * 0.5)
    cz, sz = tf.cos(rz * 0.5), tf.sin(rz * 0.5)

    # For extrinsic XYZ: R = Rx(rx) @ Ry(ry) @ Rz(rz)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz

    return tf.stack([qx, qy, qz, qw], axis=-1)


def quaternion_to_euler(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (xyzw) to extrinsic XYZ Euler angles.

    Args:
        quat: (..., 4) quaternion in xyzw order

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw]
    """
    rot_matrix = quaternion_to_rotation_matrix(quat)
    return rotation_matrix_to_euler(rot_matrix)


def quaternion_to_rotation_matrix(quat: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (xyzw) to rotation matrix.

    Args:
        quat: (..., 4) quaternion in xyzw order

    Returns:
        (..., 3, 3) rotation matrix
    """
    eps = tf.constant(EPSILON, quat.dtype)
    qx, qy, qz, qw = tf.unstack(quat, num=4, axis=-1)
    norm = tf.sqrt(tf.maximum(qw**2 + qx**2 + qy**2 + qz**2, eps))
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Rotation matrix elements
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx * qy - qw * qz)
    r13 = 2 * (qx * qz + qw * qy)

    r21 = 2 * (qx * qy + qw * qz)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy * qz - qw * qx)

    r31 = 2 * (qx * qz - qw * qy)
    r32 = 2 * (qy * qz + qw * qx)
    r33 = 1 - 2 * (qx**2 + qy**2)

    return tf.stack(
        [
            tf.stack([r11, r12, r13], axis=-1),
            tf.stack([r21, r22, r23], axis=-1),
            tf.stack([r31, r32, r33], axis=-1),
        ],
        axis=-2,
    )


def rotation_matrix_to_quaternion(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotation matrix to quaternion (xyzw).

    Args:
        rot_matrix: (..., 3, 3) rotation matrix

    Returns:
        (..., 4) quaternion in xyzw order
    """
    eps = tf.constant(EPSILON, rot_matrix.dtype)
    r11, r12, r13 = tf.unstack(rot_matrix[..., 0, :], num=3, axis=-1)
    r21, r22, r23 = tf.unstack(rot_matrix[..., 1, :], num=3, axis=-1)
    r31, r32, r33 = tf.unstack(rot_matrix[..., 2, :], num=3, axis=-1)

    trace = r11 + r22 + r33
    cond1 = trace > 0
    cond2 = tf.logical_and(r11 > r22, r11 > r33)
    cond3 = r22 > r33

    s1 = tf.sqrt(tf.maximum(trace + 1.0, eps)) * 2.0
    s2 = tf.sqrt(tf.maximum(1.0 + r11 - r22 - r33, eps)) * 2.0
    s3 = tf.sqrt(tf.maximum(1.0 + r22 - r11 - r33, eps)) * 2.0
    s4 = tf.sqrt(tf.maximum(1.0 + r33 - r11 - r22, eps)) * 2.0

    qw1 = 0.25 * s1
    qx1 = (r32 - r23) / (s1 + eps)
    qy1 = (r13 - r31) / (s1 + eps)
    qz1 = (r21 - r12) / (s1 + eps)

    qw2 = (r32 - r23) / (s2 + eps)
    qx2 = 0.25 * s2
    qy2 = (r12 + r21) / (s2 + eps)
    qz2 = (r13 + r31) / (s2 + eps)

    qw3 = (r13 - r31) / (s3 + eps)
    qx3 = (r12 + r21) / (s3 + eps)
    qy3 = 0.25 * s3
    qz3 = (r23 + r32) / (s3 + eps)

    qw4 = (r21 - r12) / (s4 + eps)
    qx4 = (r13 + r31) / (s4 + eps)
    qy4 = (r23 + r32) / (s4 + eps)
    qz4 = 0.25 * s4

    qw = tf.where(cond1, qw1, tf.where(cond2, qw2, tf.where(cond3, qw3, qw4)))
    qx = tf.where(cond1, qx1, tf.where(cond2, qx2, tf.where(cond3, qx3, qx4)))
    qy = tf.where(cond1, qy1, tf.where(cond2, qy2, tf.where(cond3, qy3, qy4)))
    qz = tf.where(cond1, qz1, tf.where(cond2, qz2, tf.where(cond3, qz3, qz4)))

    # Normalize to eliminate residual numeric drift
    q = tf.stack([qx, qy, qz, qw], axis=-1)
    q = q / (tf.norm(q, axis=-1, keepdims=True) + eps)
    return q


# =============================================================================
# 6D Rotation Representation (R6)
# =============================================================================


def rotation_matrix_to_r6(rot_matrix: tf.Tensor) -> tf.Tensor:
    """Convert rotation matrix to 6D rotation representation.

    The R6 representation contains the first two rows of the 3x3 rotation matrix:
    [r11, r12, r13, r21, r22, r23]

    Args:
        rot_matrix: (..., 3, 3) rotation matrix

    Returns:
        (..., 6) R6 representation
    """
    upper_two_rows = rot_matrix[..., :2, :]  # [..., 2, 3]
    return tf.reshape(upper_two_rows, tf.concat([tf.shape(rot_matrix)[:-2], [6]], axis=0))


def r6_to_rotation_matrix(r6: tf.Tensor) -> tf.Tensor:
    """Convert 6D rotation representation to orthonormal rotation matrix.

    Uses Gram-Schmidt orthonormalization to reconstruct valid rotation matrix.

    Args:
        r6: (..., 6) R6 representation

    Returns:
        (..., 3, 3) rotation matrix
    """
    dtype = r6.dtype
    eps = tf.constant(EPSILON, dtype)

    def _normalize(vec: tf.Tensor) -> tf.Tensor:
        norm = tf.maximum(tf.norm(vec, axis=-1, keepdims=True), eps)
        return vec / norm

    r1 = r6[..., :3]
    r2 = r6[..., 3:]

    r1 = _normalize(r1)
    # Gram-Schmidt: remove component of r2 along r1 before normalizing
    r2 = r2 - tf.reduce_sum(r2 * r1, axis=-1, keepdims=True) * r1
    r2 = _normalize(r2)

    r3 = tf.linalg.cross(r1, r2)
    r3 = _normalize(r3)

    return tf.stack([r1, r2, r3], axis=-2)


def euler_to_r6(euler: tf.Tensor) -> tf.Tensor:
    """Convert extrinsic XYZ Euler angles to 6D rotation representation.

    Args:
        euler: (..., 3) tensor of Euler angles [roll, pitch, yaw]

    Returns:
        (..., 6) R6 representation
    """
    R = euler_to_rotation_matrix(euler)
    return rotation_matrix_to_r6(R)


def r6_to_euler(r6: tf.Tensor) -> tf.Tensor:
    """Convert 6D rotation representation to extrinsic XYZ Euler angles.

    Args:
        r6: (..., 6) R6 representation

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw]
    """
    R = r6_to_rotation_matrix(r6)
    return rotation_matrix_to_euler(R)


# =============================================================================
# Coordinate Frame Transformations
# =============================================================================


def apply_coordinate_transform(
    movement_actions: tf.Tensor,
    transform_matrix: tf.Tensor,
) -> tf.Tensor:
    """Apply a coordinate frame transformation to movement actions.

    This is a general function for transforming actions from one coordinate
    frame to another. The transform_matrix defines the relationship between
    the frames: x' = C @ x for translations and R' = C @ R @ C^T for rotations.

    Args:
        movement_actions: (..., 6) tensor where [:3] = translation deltas (xyz),
                         [3:6] = Euler angles in extrinsic XYZ (roll, pitch, yaw)
        transform_matrix: (3, 3) coordinate transformation matrix C

    Returns:
        Transformed actions (..., 6) in the new coordinate frame
    """
    movement_actions = tf.convert_to_tensor(movement_actions)
    dtype = movement_actions.dtype
    C = tf.cast(transform_matrix, dtype)

    # Transform translations: t' = C @ t
    t = movement_actions[..., :3]
    t_prime = tf.linalg.matvec(C, t)

    # Transform rotations: R' = C @ R @ C^T
    euler = movement_actions[..., 3:6]
    R = euler_to_rotation_matrix(euler)
    CT = tf.linalg.matrix_transpose(C)
    R_prime = tf.linalg.matmul(tf.linalg.matmul(C, R), CT)
    euler_prime = rotation_matrix_to_euler(R_prime)

    return tf.concat([t_prime, euler_prime], axis=-1)


# Pre-defined coordinate transformation matrices
# BCZ: x'=-y, y'=-x, z'=-z
TRANSFORM_BCZ = tf.constant([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=tf.float32)

# DOBBE: x'=y, y'=-x, z'=z
TRANSFORM_DOBBE = tf.constant([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)

# JACO: x'=-y, y'=x, z'=z
TRANSFORM_JACO = tf.constant([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)


@tf.function
def coordinate_transform_bcz(movement_actions: tf.Tensor) -> tf.Tensor:
    """Transform for BCZ dataset: x'=-y, y'=-x, z'=-z."""
    return apply_coordinate_transform(movement_actions, TRANSFORM_BCZ)


@tf.function
def coordinate_transform_dobbe(movement_actions: tf.Tensor) -> tf.Tensor:
    """Transform for DOBBE dataset: x'=y, y'=-x, z'=z."""
    return apply_coordinate_transform(movement_actions, TRANSFORM_DOBBE)


@tf.function
def coordinate_transform_jaco(movement_actions: tf.Tensor) -> tf.Tensor:
    """Transform for JACO dataset: x'=-y, y'=x, z'=z."""
    return apply_coordinate_transform(movement_actions, TRANSFORM_JACO)


# =============================================================================
# Euler Angle Utilities
# =============================================================================


@tf.function
def euler_diff(angles1: tf.Tensor, angles2: tf.Tensor) -> tf.Tensor:
    """Compute relative Euler angle difference.

    Computes angles_rel such that R(angles2) @ R(angles_rel) = R(angles1)

    Args:
        angles1: (..., 3) Euler angles [roll, pitch, yaw] (extrinsic XYZ)
        angles2: (..., 3) Euler angles [roll, pitch, yaw] (extrinsic XYZ)

    Returns:
        (..., 3) relative Euler angles [roll, pitch, yaw]
    """
    R1 = euler_to_rotation_matrix(angles1)
    R2 = euler_to_rotation_matrix(angles2)

    # Compute relative rotation: Rrel = R2^T @ R1
    Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)

    return rotation_matrix_to_euler(Rrel)


@tf.function
def zxy_to_xyz(angles: tf.Tensor, degrees: bool = False) -> tf.Tensor:
    """Convert intrinsic Z-X-Y Euler angles to extrinsic X-Y-Z Euler angles.

    Args:
        angles: (..., 3) tensor of (az, ax, ay) in radians (intrinsic ZXY)
        degrees: Whether input/output are in degrees

    Returns:
        (..., 3) tensor of (roll, pitch, yaw) in extrinsic XYZ
    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if degrees:
        angles = angles * (_tf_pi() / 180.0)

    az, ax, ay = angles[..., 0], angles[..., 1], angles[..., 2]

    # Build rotation for intrinsic "zxy": R = Rz(az) @ Rx(ax) @ Ry(ay)
    Rz = rot_z(az)
    Rx = rot_x(ax)
    Ry = rot_y(ay)
    R = tf.linalg.matmul(tf.linalg.matmul(Rz, Rx), Ry)

    out = rotation_matrix_to_euler(R)
    if degrees:
        out = out * (180.0 / _tf_pi())
    return out


@tf.function
def matrix_to_xyzrpy(T: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Extract position and Euler angles from 4x4 transformation matrix.

    Args:
        T: (..., 4, 4) homogeneous transformation matrix
        eps: Epsilon for gimbal lock handling

    Returns:
        (..., 6) tensor of [x, y, z, roll, pitch, yaw] (extrinsic XYZ)
    """
    T = tf.convert_to_tensor(T)
    xyz = T[..., :3, 3]
    R = T[..., :3, :3]
    rpy = rotation_matrix_to_euler(R, eps=eps)
    return tf.concat([xyz, rpy], axis=-1)


# =============================================================================
# NumPy Versions (for non-TF contexts)
# =============================================================================


def euler_xyz_to_rot_np(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build rotation matrix from XYZ extrinsic rotations (NumPy version)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rx @ Ry @ Rz


# =============================================================================
# Axis-Angle Conversions
# =============================================================================


def axis_angle_to_r6(axis_angle: tf.Tensor) -> tf.Tensor:
    """Convert axis-angle representation to 6D rotation representation.

    Args:
        axis_angle: (..., 3) tensor where magnitude is angle and direction is axis

    Returns:
        (..., 6) R6 representation
    """
    import tensorflow_graphics.geometry.transformation as tft

    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = tf.where(angle < EPSILON, tf.ones_like(angle), angle)
    axis = axis_angle / safe_angle
    axis = tf.where(angle < EPSILON, tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype), axis)

    rotation_matrix = tft.rotation_matrix_3d.from_axis_angle(axis, angle)
    return tf.concat([rotation_matrix[..., 0, :], rotation_matrix[..., 1, :]], axis=-1)


def axis_angle_to_euler(axis_angle: tf.Tensor) -> tf.Tensor:
    """Convert axis-angle representation to extrinsic XYZ Euler angles.

    Args:
        axis_angle: (..., 3) tensor where magnitude is angle and direction is axis

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw]
    """
    import tensorflow_graphics.geometry.transformation as tft

    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = tf.where(angle < EPSILON, tf.ones_like(angle), angle)
    axis = axis_angle / safe_angle
    axis = tf.where(angle < EPSILON, tf.constant([1.0, 0.0, 0.0], dtype=axis.dtype), axis)

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


def wxyz_to_r6(quaternion: tf.Tensor) -> tf.Tensor:
    """Convert quaternion (wxyz format) to 6D rotation representation.

    Args:
        quaternion: (..., 4) quaternion in [w, x, y, z] format

    Returns:
        (..., 6) R6 representation
    """
    import tensorflow_graphics.geometry.transformation as tft

    # Convert from [w, x, y, z] to [x, y, z, w]
    quat_xyzw = tf.concat([quaternion[..., 1:4], quaternion[..., 0:1]], axis=-1)
    rotation_matrix = tft.rotation_matrix_3d.from_quaternion(quat_xyzw)
    return tf.concat([rotation_matrix[..., 0, :], rotation_matrix[..., 1, :]], axis=-1)
