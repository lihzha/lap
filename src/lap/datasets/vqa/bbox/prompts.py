"""Prompt templates for bounding box VQA datasets.

This module contains all prompt templates used for bbox detection and
direction classification tasks across robot VQA datasets.
"""

import tensorflow as tf

# =============================================================================
# GENERAL BBOX DETECTION PROMPTS
# =============================================================================

GENERAL_BBOX_PROMPT_PARTS: list[tuple[str, str]] = [
    ("Show me where the robot should move its end-effector to reach the ", " in the image."),
    ("Describe the location the robot should align its gripper with to reach the ", " in the image."),
    ("Locate the region where the robot should position its wrist to interact with the ", " in the image."),
    ("Mark the location the robot should target with its gripper to reach the ", "."),
    ("Identify the spot the robot should move its arm toward to approach the ", "."),
    ("Find the region the robot should align its end-effector with to reach the ", " in the image."),
    ("Highlight the area the robot should approach with its manipulator to reach the ", " in the image."),
    ("Show me where the robot would position its gripper to approach the ", " in the image."),
    ("Indicate where the robot should move its arm to reach the ", "."),
    ("Mark the location the robot should target to interact with the ", "."),
    ("Highlight the region the robot should move toward to grasp the ", "."),
    ("Identify where the robot should position its wrist relative to the ", "."),
    ("Point out the spot the robot would navigate its arm to in order to reach the ", "."),
    ("Locate where the robot would need to move its end-effector to get closer to the ", " in the image."),
    ("Pinpoint the position the robot should move its gripper toward to access the ", "."),
    ("Show the area the robot should aim its arm toward when approaching the ", "."),
    ("Outline the region that would guide the robot's end-effector toward the ", "."),
    # Manipulation-focused prompts
    ("Indicate the exact region a robot should target with its gripper when reaching for the ", "."),
    ("Highlight the bounding region the robot should aim its wrist toward to reach the ", "."),
    ("Mark the precise location where the robot should position its end-effector to approach the ", "."),
    ("Identify the spatial region where the robot would place its gripper to interact with the ", "."),
    ("Show the area the robot should move its arm into to reach the ", "."),
    ("Locate the target region the robot should align its manipulator with to access the ", "."),
    ("Point out the position the robot would need to occupy with its wrist to manipulate the ", "."),
    ("Outline the region that represents the robot's goal location for reaching the ", "."),
    ("Find the area in the image that the robot should move its end-effector toward to reach the ", "."),
    ("Mark the destination region a robot should select with its gripper to successfully approach the ", "."),
]

# =============================================================================
# ROBOT BBOX PROMPT CONSTRUCTION
# =============================================================================

# Base action phrases (part 1) for robot bbox prompts
_ROBOT_BBOX_PART1: list[str] = [
    "Pick up the ",
    "Grasp the ",
    "Move near to the ",
    "Navigate to the ",
]

# Part 2 variants for different coordinate frames
_ROBOT_BBOX_PART2_IMAGE: list[str] = [
    ", predict where it is in the image.",
    ", show where it is in the image.",
    ", locate it in the image.",
    ", find it in the image.",
]

_ROBOT_BBOX_PART2_ROBOT_BASE: list[str] = [
    ", predict where it is in the robot base frame.",
    ", relative to the robot base.",
    ", with respect to the robot base.",
    ", looking from the external camera.",
]

_ROBOT_BBOX_PART2_EE: list[str] = [
    ", predict where it is in the end-effector frame.",
    ", with respect to the robot gripper.",
    ", relative to the end-effector.",
    ", in the wrist camera.",
    ", looking from the wrist camera.",
]

# Full combination: all part1 x all part2 variants
ROBOT_BBOX_PROMPT_PARTS: list[tuple[str, str]] = [
    (p1, p2)
    for p1 in _ROBOT_BBOX_PART1
    for p2 in _ROBOT_BBOX_PART2_IMAGE + _ROBOT_BBOX_PART2_ROBOT_BASE + _ROBOT_BBOX_PART2_EE
] + GENERAL_BBOX_PROMPT_PARTS

# OXE-specific: robot base frame + image (no end-effector frame)
ROBOT_BBOX_PROMPT_PARTS_OXE: list[tuple[str, str]] = [
    (p1, p2)
    for p1 in _ROBOT_BBOX_PART1
    for p2 in _ROBOT_BBOX_PART2_IMAGE + _ROBOT_BBOX_PART2_ROBOT_BASE
] + GENERAL_BBOX_PROMPT_PARTS

# End-effector specific: end-effector frame + image (for DROID, PACO, LVIS, etc.)
ROBOT_BBOX_PROMPT_PARTS_EE: list[tuple[str, str]] = [
    (p1, p2)
    for p1 in _ROBOT_BBOX_PART1
    for p2 in _ROBOT_BBOX_PART2_IMAGE + _ROBOT_BBOX_PART2_EE
] + GENERAL_BBOX_PROMPT_PARTS

# =============================================================================
# DIRECTION CLASSIFICATION PROMPTS
# =============================================================================

DIRECTION_PROMPT_PARTS: list[tuple[str, str]] = [
    ("From the image center, imagine the robot moving its end-effector toward the ", " and predict the direction."),
    ("Relative to the center of the image, imagine the robot aligning its arm toward the ", " and describe the movement direction."),
    ("If the robot's base were at the center of the image, which way would the arm extend to reach the ", "."),
    ("Looking from the center of the frame, imagine the robot orienting its gripper toward the ", " and state the direction."),
    ("Which direction from the center would the robot move its end-effector to reach the ", " in this image."),
    ("Imagine the robot must reposition its arm to interact with the ", " and describe its direction."),
    ("Describe which direction the robot would move its gripper to approach the ", " in the image."),
    ("Describe the direction the robot's arm should sweep to align with the ", " in the image."),
    ("Point out the direction the robot should move its end-effector to reach the ", "."),
    ("Show me where the robot should aim its arm to reach the ", "."),
    ("Describe where the robot would move its wrist to reach the ", " relative to the center of the image."),
    ("Show me the direction the robot should move its arm toward the ", " relative to the center of the image."),
    ("Imagine the robot needs to extend its arm toward the ", " and predict the direction."),
    ("Imagine the robot needs to reposition its manipulator to the ", " and predict the direction."),
    ("If the robot needs to grasp the ", ", predict the direction it would move its arm."),
    # Manipulation-focused prompts
    ("From the image center, predict the direction the robot should move its end-effector to make contact with the ", "."),
    ("Assuming the robot starts with its gripper at the image center, describe the direction it should move toward the ", "."),
    ("If the robot had to plan a straight-line reach from the center to the ", ", which direction would the arm move."),
    ("Imagine the robot is positioned at the center and must align its gripper with the ", "; indicate the direction."),
    ("From the center of the image, in which direction should the robot move its wrist to approach the ", "."),
    ("If the robot were planning a pre-grasp motion from the center, describe the direction toward the ", "."),
    ("Predict the initial arm movement direction a robot would take from the center to reach the ", "."),
    ("Considering a robot at the center, which direction would it orient its gripper to approach the ", "."),
    ("From a manipulation standpoint, which direction should the robot move its arm from the center to reach the ", "."),
    ("If the robot plans a direct reach from the center to the ", ", what direction would the end-effector move."),
]

# =============================================================================
# ROBOT DIRECTION PROMPT CONSTRUCTION
# =============================================================================

_ROBOT_DIRECTION_PART1: list[str] = [
    "Pick up the ",
    "Move to the ",
    "Grab the ",
    "Navigate to the ",
]

_ROBOT_DIRECTION_PART2_EE: list[str] = [
    ", predict the robot's action in the end-effector frame.",
    ", with respect to the robot gripper.",
    ", relative to the end-effector.",
    ", in the wrist camera.",
    ", looking from the wrist camera.",
]

_ROBOT_DIRECTION_PART2_ROBOT_BASE: list[str] = [
    ", predict the robot's action in the robot base frame.",
    ", relative to the robot base.",
    ", with respect to the robot base.",
    ", in the robot base coordinate frame.",
    ", in the robot base frame.",
    ", looking from the external camera.",
]

# OXE-specific: robot base frame only
ROBOT_DIRECTION_PROMPT_PARTS_OXE: list[tuple[str, str]] = [
    (p1, p2)
    for p1 in _ROBOT_DIRECTION_PART1
    for p2 in _ROBOT_DIRECTION_PART2_ROBOT_BASE
] + DIRECTION_PROMPT_PARTS

# End-effector specific: end-effector frame only (for DROID, etc.)
ROBOT_DIRECTION_PROMPT_PARTS_EE: list[tuple[str, str]] = [
    (p1, p2)
    for p1 in _ROBOT_DIRECTION_PART1
    for p2 in _ROBOT_DIRECTION_PART2_EE
] + DIRECTION_PROMPT_PARTS


# =============================================================================
# PROMPT SAMPLING HELPER
# =============================================================================

def sample_prompt_tf(
    prompt_parts: list[tuple[str, str]],
    category_name: tf.Tensor,
    seed_pair: tuple[int, tf.Tensor],
) -> tf.Tensor:
    """Sample a random prompt template and fill in the category name.

    Args:
        prompt_parts: List of (prefix, suffix) tuples for prompt templates.
        category_name: tf.string tensor with the object category/label.
        seed_pair: Tuple of (base_seed, hash_value) for stateless random.

    Returns:
        tf.string tensor with the complete prompt.
    """
    num_prompts = len(prompt_parts)
    prompt_idx = tf.random.stateless_uniform(
        [], seed=seed_pair, minval=0, maxval=num_prompts, dtype=tf.int32
    )

    # Create branches for each prompt template
    def make_prompt_fn(idx):
        prefix, suffix = prompt_parts[idx]

        def fn():
            return tf.strings.join([prefix, category_name, suffix])

        return fn

    prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
    return tf.switch_case(prompt_idx, branch_fns=prompt_branches)
