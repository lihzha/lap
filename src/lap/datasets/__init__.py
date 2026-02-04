"""RLDS datasets package.

This package provides dataset classes for loading and processing
robotics and VQA datasets in a standardized format.

Usage:
    from lap.datasets import get_dataset_class, register_dataset
    from lap.datasets import TrajectoryOutputBuilder, ObservationBuilder
    from lap.datasets import get_vqa_dataset_id, get_dataset_config
"""

# Base classes
from lap.datasets.base_dataset import BaseDataset
from lap.datasets.base_dataset import BaseRobotDataset
from lap.datasets.output_schema import ObservationBuilder
from lap.datasets.output_schema import TrajectoryOutputBuilder
from lap.datasets.registry import DATASET_REGISTRY  # Registry
from lap.datasets.registry import VQA_DATASET_NAMES
from lap.datasets.registry import WRIST_ROTATION_PATTERNS
from lap.datasets.registry import DatasetConfig  # Configuration (now in registry)
from lap.datasets.registry import DatasetMetadata
from lap.datasets.registry import get_action_bounds
from lap.datasets.registry import get_dataset_class
from lap.datasets.registry import get_dataset_class_with_fallback
from lap.datasets.registry import get_dataset_config
from lap.datasets.registry import get_dataset_metadata
from lap.datasets.registry import get_num_vqa_datasets
from lap.datasets.registry import get_tfds_name_with_version
from lap.datasets.registry import get_vqa_dataset_id
from lap.datasets.registry import get_vqa_dataset_name
from lap.datasets.registry import is_bimanual_dataset
from lap.datasets.registry import is_navigation_dataset
from lap.datasets.registry import is_vqa_dataset
from lap.datasets.registry import list_registered_datasets
from lap.datasets.registry import needs_wrist_rotation
from lap.datasets.registry import register_dataset
from lap.datasets.registry import register_dataset_config
from lap.datasets.registry import requires_hash_tables

__all__ = [
    # Registry
    "DATASET_REGISTRY",
    "VQA_DATASET_NAMES",
    "DatasetMetadata",
    "get_dataset_class",
    "get_dataset_class_with_fallback",
    "get_dataset_metadata",
    "get_num_vqa_datasets",
    "get_vqa_dataset_id",
    "get_vqa_dataset_name",
    "is_vqa_dataset",
    "list_registered_datasets",
    "register_dataset",
    "requires_hash_tables",
    # Configuration
    "DatasetConfig",
    "get_action_bounds",
    "get_dataset_config",
    "get_tfds_name_with_version",
    "is_bimanual_dataset",
    "is_navigation_dataset",
    "needs_wrist_rotation",
    "register_dataset_config",
    "WRIST_ROTATION_PATTERNS",
    # Output builders
    "ObservationBuilder",
    "TrajectoryOutputBuilder",
    # Base classes
    "BaseDataset",
    "BaseRobotDataset",
]
