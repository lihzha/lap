"""Shared constants for RLDS datasets."""

import tensorflow as tf

from lap.datasets.registry import WRIST_ROTATION_PATTERNS

# Fallback instructions for datasets with empty language annotations
FALLBACK_INSTRUCTIONS = tf.constant(
    [
        "Do something useful.",
        "Complete the task.",
        "Perform the task.",
        "Carry out the objective.",
        "Execute the current task.",
        "Accomplish the goal.",
        "Proceed with the task.",
        "Handle the task at hand.",
        "Continue the operation.",
        "Fulfill the task.",
        "Take meaningful steps.",
        "Demonstrate useful behavior.",
        "Act in a useful manner.",
        "Engage in productive actions.",
        "Make useful moves.",
        "Undertake useful actions.",
        "Behave purposefully.",
        "Start the activity.",
    ],
    dtype=tf.string,
)

# Number of fallback instructions (for hash bucket operations)
NUM_FALLBACK_INSTRUCTIONS = len(FALLBACK_INSTRUCTIONS)

# Datasets that need wrist camera rotation by 180 degrees.
# Backward-compatible alias derived from the registry source of truth.
DATASETS_REQUIRING_WRIST_ROTATION: set[str] = {
    *WRIST_ROTATION_PATTERNS,
}

# Default image resolution for preprocessing
DEFAULT_IMAGE_RESOLUTION = (224, 224)

# Gripper state thresholds
GRIPPER_OPEN_THRESHOLD = 0.95
GRIPPER_CLOSE_THRESHOLD = 0.05
GRIPPER_BINARIZE_THRESHOLD = 0.5

# Small epsilon for numerical stability
EPSILON = 1e-8
