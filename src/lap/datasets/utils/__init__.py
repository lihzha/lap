"""Dataset utilities for RLDS datasets.

This package provides utilities for working with robot learning datasets
including transforms, configurations, image processing, and rotation utilities.
"""

# Re-export commonly used items for convenience
# Dataset configs and mixtures
from lap.datasets.utils.configs import OXE_DATASET_CONFIGS
from lap.datasets.utils.configs import OXE_DATASET_METADATA

# Constants
from lap.datasets.utils.constants import DATASETS_REQUIRING_WRIST_ROTATION
from lap.datasets.utils.constants import DEFAULT_IMAGE_RESOLUTION
from lap.datasets.utils.constants import EPSILON
from lap.datasets.utils.constants import FALLBACK_INSTRUCTIONS
from lap.datasets.utils.constants import GRIPPER_BINARIZE_THRESHOLD
from lap.datasets.utils.constants import GRIPPER_OPEN_THRESHOLD
from lap.datasets.utils.dataset_discovery import ensure_datasets_registered

# Encoding types
from lap.datasets.utils.helpers import ActionEncoding
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import StateEncoding
from lap.datasets.utils.helpers import state_encoding_to_type
from lap.datasets.utils.image_utils import make_decode_images_fn
from lap.datasets.utils.image_utils import tf_maybe_rotate_180

# Image utilities
from lap.datasets.utils.image_utils import tf_rotate_180
from lap.datasets.utils.mixtures import OXE_NAMED_MIXTURES
from lap.datasets.utils.normalization_and_config import allocate_threads
from lap.datasets.utils.normalization_and_config import load_dataset_kwargs

# Data utilities
from lap.datasets.utils.normalization_and_config import normalize_action_and_proprio
from lap.datasets.utils.normalization_and_config import pprint_data_mixture
from lap.datasets.utils.rotation_utils import apply_coordinate_transform
from lap.datasets.utils.rotation_utils import axis_angle_to_euler
from lap.datasets.utils.rotation_utils import axis_angle_to_r6
from lap.datasets.utils.rotation_utils import coordinate_transform_bcz
from lap.datasets.utils.rotation_utils import coordinate_transform_dobbe
from lap.datasets.utils.rotation_utils import coordinate_transform_jaco
from lap.datasets.utils.rotation_utils import euler_diff
from lap.datasets.utils.rotation_utils import euler_to_quaternion
from lap.datasets.utils.rotation_utils import euler_to_r6

# Rotation utilities
from lap.datasets.utils.rotation_utils import euler_to_rotation_matrix
from lap.datasets.utils.rotation_utils import matrix_to_xyzrpy
from lap.datasets.utils.rotation_utils import quaternion_to_euler
from lap.datasets.utils.rotation_utils import quaternion_to_rotation_matrix
from lap.datasets.utils.rotation_utils import r6_to_euler
from lap.datasets.utils.rotation_utils import r6_to_rotation_matrix
from lap.datasets.utils.rotation_utils import rotation_matrix_to_euler
from lap.datasets.utils.rotation_utils import rotation_matrix_to_quaternion
from lap.datasets.utils.rotation_utils import rotation_matrix_to_r6
from lap.datasets.utils.rotation_utils import wxyz_to_r6
from lap.datasets.utils.rotation_utils import zxy_to_xyz

# Statistics and discovery utilities
from lap.datasets.utils.statistics import GlobalStatisticsBuilder
from lap.datasets.utils.tfdata_pipeline import dataset_size
from lap.datasets.utils.tfdata_pipeline import gather_with_last_value_padding

# Dataset utilities
from lap.datasets.utils.tfdata_pipeline import gather_with_padding
from lap.datasets.utils.tfdata_pipeline import prepare_batched_dataset
from lap.datasets.utils.transform_helpers import binarize_gripper_actions
from lap.datasets.utils.transform_helpers import build_matrix_state_transform
from lap.datasets.utils.transform_helpers import build_standard_eef_transform

# Transform helpers
from lap.datasets.utils.transform_helpers import compute_padded_movement_actions
from lap.datasets.utils.transform_helpers import extract_state_from_matrix
from lap.datasets.utils.transform_helpers import fill_empty_language_instruction
from lap.datasets.utils.transform_helpers import invert_gripper_actions
from lap.datasets.utils.transform_helpers import rel2abs_gripper_actions
from lap.datasets.utils.transform_helpers import rescale_action_with_bound

# Transform registry
from lap.datasets.utils.transforms import OXE_STANDARDIZATION_TRANSFORMS

__all__ = [
    # Mixtures
    "OXE_NAMED_MIXTURES",
    # Constants
    "DATASETS_REQUIRING_WRIST_ROTATION",
    "FALLBACK_INSTRUCTIONS",
    "DEFAULT_IMAGE_RESOLUTION",
    "GRIPPER_OPEN_THRESHOLD",
    "GRIPPER_BINARIZE_THRESHOLD",
    "EPSILON",
    # Encoding types
    "ActionEncoding",
    "StateEncoding",
    "NormalizationType",
    "state_encoding_to_type",
    # Configs
    "OXE_DATASET_CONFIGS",
    "OXE_DATASET_METADATA",
    # Rotation utilities
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "rotation_matrix_to_r6",
    "r6_to_rotation_matrix",
    "euler_to_r6",
    "r6_to_euler",
    "apply_coordinate_transform",
    "coordinate_transform_bcz",
    "coordinate_transform_dobbe",
    "coordinate_transform_jaco",
    "euler_diff",
    "zxy_to_xyz",
    "matrix_to_xyzrpy",
    "axis_angle_to_r6",
    "axis_angle_to_euler",
    "wxyz_to_r6",
    # Transform helpers
    "compute_padded_movement_actions",
    "extract_state_from_matrix",
    "fill_empty_language_instruction",
    "binarize_gripper_actions",
    "invert_gripper_actions",
    "rel2abs_gripper_actions",
    "build_standard_eef_transform",
    "build_matrix_state_transform",
    "rescale_action_with_bound",
    # Image utilities
    "tf_rotate_180",
    "tf_maybe_rotate_180",
    "make_decode_images_fn",
    # Data utilities
    "normalize_action_and_proprio",
    "load_dataset_kwargs",
    "pprint_data_mixture",
    "allocate_threads",
    # Dataset utilities
    "gather_with_padding",
    "gather_with_last_value_padding",
    "dataset_size",
    "prepare_batched_dataset",
    # Transforms
    "OXE_STANDARDIZATION_TRANSFORMS",
    # Statistics
    "GlobalStatisticsBuilder",
    "ensure_datasets_registered",
]
