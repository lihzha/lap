"""Transform helper functions for reducing code duplication in dataset transforms.

This module provides reusable building blocks for dataset-specific transforms,
including state extraction, action computation, and language instruction handling.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from lap.datasets.utils.constants import FALLBACK_INSTRUCTIONS
from lap.datasets.utils.constants import NUM_FALLBACK_INSTRUCTIONS
from lap.datasets.utils.rotation_utils import euler_diff
from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy

# =============================================================================
# Action Computation Helpers
# =============================================================================


def compute_padded_movement_actions(eef_state: tf.Tensor) -> tf.Tensor:
    """Compute movement actions as deltas between consecutive states.

    Follows the DROID convention where action[t] = state[t+1] - state[t],
    and action[T-1] = 0 (zero-padded). Preserves trajectory length.

    Args:
        eef_state: End-effector state tensor (..., T, 6) containing [xyz, rpy]

    Returns:
        Padded movement actions (..., T, 6) containing [delta_xyz, delta_rpy]
    """
    # Compute deltas: action[t] = state[t+1] - state[t]
    movement_actions = tf.concat(
        (
            eef_state[1:, :3] - eef_state[:-1, :3],  # Position deltas
            euler_diff(eef_state[1:, 3:6], eef_state[:-1, 3:6]),  # Rotation deltas
        ),
        axis=-1,
    )
    # Pad with zeros for the last timestep
    padded = tf.concat(
        [movement_actions, tf.zeros((1, tf.shape(movement_actions)[1]))],
        axis=0,
    )
    return padded


# =============================================================================
# State Extraction Helpers
# =============================================================================


def extract_state_from_matrix(
    state_matrix_flat: tf.Tensor,
    gripper_state: tf.Tensor,
    gripper_scale: float = 0.079,
) -> tf.Tensor:
    """Extract EEF state from column-major flattened transformation matrix.

    Common pattern used in austin_buds, austin_sailor, austin_sirius, viola,
    utaustin_mutex datasets.

    Args:
        state_matrix_flat: (..., 16) flattened 4x4 transformation matrix (column-major)
        gripper_state: (..., 1) gripper state values
        gripper_scale: Scale factor for gripper normalization

    Returns:
        (..., 7) state tensor [x, y, z, roll, pitch, yaw, gripper]
    """
    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(state_matrix_flat, [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])

    xyzrpy = matrix_to_xyzrpy(state_matrix)
    normalized_gripper = tf.clip_by_value(gripper_state / gripper_scale, 0, 1)

    return tf.concat([xyzrpy, normalized_gripper], axis=-1)


# =============================================================================
# Language Instruction Helpers
# =============================================================================


def fill_empty_language_instruction(
    trajectory: dict[str, Any],
    use_deterministic_fallback: bool = True,
) -> dict[str, Any]:
    """Fill empty language instructions with fallback text.

    Args:
        trajectory: Trajectory dictionary with 'language_instruction' key
        use_deterministic_fallback: If True, use deterministic index based on
                                   state hash. If False, use random selection.

    Returns:
        Trajectory with filled language instructions
    """
    traj_len = tf.shape(trajectory["language_instruction"])[0]
    instruction = trajectory["language_instruction"][0]

    is_empty = tf.logical_or(
        tf.equal(tf.strings.length(tf.strings.strip(instruction)), 0),
        tf.equal(instruction, tf.constant("", dtype=tf.string)),
    )

    if use_deterministic_fallback:
        # Use deterministic index based on hash of first observation state
        state_hash = tf.strings.to_hash_bucket_fast(
            tf.strings.as_string(tf.reduce_sum(trajectory["observation"]["state"][0])),
            NUM_FALLBACK_INSTRUCTIONS,
        )
        fallback = FALLBACK_INSTRUCTIONS[state_hash]
    else:
        # Use random selection
        fallback = tf.random.shuffle(FALLBACK_INSTRUCTIONS)[0]

    selected = tf.cond(is_empty, lambda: fallback, lambda: instruction)
    trajectory["language_instruction"] = tf.fill([traj_len], selected)

    return trajectory


# =============================================================================
# Gripper Action Helpers
# =============================================================================


def binarize_gripper_actions(
    actions: tf.Tensor,
    threshold: float = 0.95,
) -> tf.Tensor:
    """Convert gripper actions from continuous to binary (0/1).

    Handles intermediate values by looking ahead to find the target state.

    Args:
        actions: Continuous gripper actions
        threshold: Threshold for open/closed classification

    Returns:
        Binary gripper actions (0 = closed, 1 = open)
    """
    open_mask = actions > threshold
    closed_mask = actions < (1 - threshold)
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: carry, lambda: is_open_float[i])

    return tf.scan(
        scan_fn,
        tf.range(tf.shape(actions)[0]),
        tf.cast(actions[-1], tf.float32),
        reverse=True,
    )


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Invert gripper actions (0 <-> 1)."""
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Convert relative gripper actions to absolute.

    Relative: +1 for closing, -1 for opening
    Absolute: 0 = closed, 1 = open

    Args:
        actions: Relative gripper actions

    Returns:
        Absolute gripper actions
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assume open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    return tf.cast(new_actions, tf.float32) / 2 + 0.5


# =============================================================================
# Common Transform Builders
# =============================================================================


def build_standard_eef_transform(
    trajectory: dict[str, Any],
    state_key: str = "state",
    state_slice: slice | None = None,
    gripper_slice: slice | None = None,
    gripper_action_slice: slice | None = None,
    invert_gripper: bool = True,
    clip_gripper: bool = True,
) -> dict[str, Any]:
    """Build a standard EEF-based transform with movement actions.

    This is a common pattern used in many dataset transforms that:
    1. Extract EEF state from observation
    2. Compute padded movement actions
    3. Create language_action and action tensors

    Args:
        trajectory: Input trajectory dictionary
        state_key: Key for state in observation dict
        state_slice: Slice for extracting EEF state (default: first 6 dims)
        gripper_slice: Slice for extracting gripper state (default: last dim)
        gripper_action_slice: Slice for gripper action (default: last dim)
        invert_gripper: Whether to invert gripper action
        clip_gripper: Whether to clip gripper to [0, 1]

    Returns:
        Transformed trajectory
    """
    # Default slices
    if state_slice is None:
        state_slice = slice(None, 6)
    if gripper_slice is None:
        gripper_slice = slice(-1, None)
    if gripper_action_slice is None:
        gripper_action_slice = slice(-1, None)

    # Extract EEF state
    eef_state = trajectory["observation"][state_key][:, state_slice]

    # Process gripper action
    gripper_action = trajectory["action"][:, gripper_action_slice]
    if clip_gripper:
        gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    if invert_gripper:
        gripper_action = invert_gripper_actions(gripper_action)

    # Compute movement actions
    padded_movement = compute_padded_movement_actions(eef_state)

    # Build outputs
    trajectory["language_action"] = tf.concat([padded_movement, gripper_action], axis=1)
    trajectory["action"] = tf.concat([eef_state, gripper_action], axis=1)

    return trajectory


def build_matrix_state_transform(
    trajectory: dict[str, Any],
    state_matrix_key: str = "state",
    matrix_slice: slice | None = None,
    gripper_key: str = "state",
    gripper_slice: slice | None = None,
    gripper_scale: float = 0.079,
    invert_gripper_action: bool = True,
    clip_gripper_action: bool = True,
    fill_empty_lang: bool = True,
    use_deterministic_fallback: bool = True,
) -> dict[str, Any]:
    """Build transform for datasets with transformation matrix state.

    Common pattern for austin_buds, austin_sailor, austin_sirius, viola,
    utaustin_mutex datasets.

    Args:
        trajectory: Input trajectory dictionary
        state_matrix_key: Key for state matrix in observation
        matrix_slice: Slice for extracting matrix (default: last 16 dims)
        gripper_key: Key for gripper state in observation
        gripper_slice: Slice for gripper state
        gripper_scale: Scale factor for gripper normalization
        invert_gripper_action: Whether to invert gripper action
        clip_gripper_action: Whether to clip gripper action to [0, 1]
        fill_empty_lang: Whether to fill empty language instructions
        use_deterministic_fallback: Use deterministic or random fallback

    Returns:
        Transformed trajectory
    """
    # Default slices
    if matrix_slice is None:
        matrix_slice = slice(-16, None)
    if gripper_slice is None:
        gripper_slice = slice(-1, None)

    # Extract state from matrix
    state_matrix_flat = trajectory["observation"][state_matrix_key][:, matrix_slice]
    gripper_state = trajectory["observation"][gripper_key][:, gripper_slice]

    trajectory["observation"]["state"] = extract_state_from_matrix(state_matrix_flat, gripper_state, gripper_scale)

    # Process gripper action
    gripper_action = trajectory["action"][:, -1:]
    if clip_gripper_action:
        gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    if invert_gripper_action:
        gripper_action = invert_gripper_actions(gripper_action)

    # Compute movement actions
    padded_movement = compute_padded_movement_actions(trajectory["observation"]["state"][:, :6])

    # Build outputs
    trajectory["language_action"] = tf.concat([padded_movement, gripper_action], axis=1)
    trajectory["action"] = tf.concat([trajectory["observation"]["state"][:, :6], gripper_action], axis=1)

    # Fill empty language instructions
    if fill_empty_lang:
        trajectory = fill_empty_language_instruction(trajectory, use_deterministic_fallback)

    return trajectory


# =============================================================================
# Action Rescaling Helpers
# =============================================================================


def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
    """Rescale actions to a target range.

    Formula: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range

    Args:
        actions: Input actions tensor
        low: Original lower bound
        high: Original upper bound
        safety_margin: Margin to add to output bounds
        post_scaling_max: Target maximum
        post_scaling_min: Target minimum

    Returns:
        Rescaled actions
    """
    resc_actions = (actions - low) / (high - low) * (post_scaling_max - post_scaling_min) + post_scaling_min
    return tf.clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )
