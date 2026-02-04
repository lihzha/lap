"""Dataset registry for dynamic dataset instantiation.

This module provides a centralized registry for dataset classes and their
configurations, enabling automatic dispatch based on dataset name patterns
without hard-coded if-elif chains.

Features:
- Exact name registration and pattern-based matching
- Automatic VQA dataset ID assignment
- Rich metadata and configuration per dataset
- Single source of truth for dataset settings

Usage:
    # Register a dataset with configuration
    @register_dataset(
        name="droid",
        config=DatasetConfig(
            needs_wrist_rotation=True,
            original_image_size=(320, 180),
        ),
        requires_hash_tables=True,
    )
    class DroidDataset(BaseRobotDataset):
        ...

    # Register a dataset with a pattern matcher
    @register_dataset(matcher=lambda n: n.startswith("libero"))
    class LiberoDataset(BaseRobotDataset):
        ...

    # Register a VQA dataset (auto-assigns ID)
    @register_dataset(name="coco_captions", is_vqa=True)
    class CocoCaption(BaseVQADataset):
        ...

    # Get the appropriate dataset class
    cls = get_dataset_class("droid")  # Returns DroidDataset
    cls = get_dataset_class("libero_goal")  # Returns LiberoDataset

    # Get dataset configuration
    config = get_dataset_config("droid")  # Returns DatasetConfig

    # Get VQA dataset ID (auto-assigned)
    vqa_id = get_vqa_dataset_id("coco_captions")  # Returns 1, 2, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lap.datasets.base_dataset import BaseRobotDataset


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a single dataset.

    This centralizes all dataset-specific settings that were previously
    scattered across multiple files.

    Attributes:
        name: Dataset name (e.g., "droid", "bridge_v2_oxe").
        tfds_version: Optional TFDS version string (e.g., "1.0.0").
        needs_wrist_rotation: Whether wrist camera images need 180° rotation.
        action_bounds: Optional (min, max) bounds for action filtering.
        original_image_size: Original image dimensions (width, height) for bbox normalization.
        frame_offset: Offset to add to frame indices for bbox lookup.
        is_bimanual: Whether this is a bimanual robot dataset.
        is_navigation: Whether this is a navigation dataset.
        force_no_wrist_image: Force disable wrist image even if available.
        control_frequency: Control frequency in Hz (for horizon computation).
    """

    name: str = ""
    tfds_version: str | None = None
    needs_wrist_rotation: bool = False
    action_bounds: tuple[float, float] | None = None
    original_image_size: tuple[int, int] = (256, 256)
    frame_offset: int = 0
    is_bimanual: bool = False
    is_navigation: bool = False
    force_no_wrist_image: bool = False
    control_frequency: int | None = None

    # Optional custom filters
    trajectory_filter: Callable | None = None
    frame_filter: Callable | None = None


# =============================================================================
# PRE-DEFINED CONFIGURATIONS
# =============================================================================

# These can be used when registering datasets
DROID_CONFIG = DatasetConfig(
    name="droid",
    needs_wrist_rotation=True,
    original_image_size=(320, 180),
)

FMB_CONFIG = DatasetConfig(
    name="fmb",
    tfds_version="1.0.0",
)

DOBBE_CONFIG = DatasetConfig(
    name="dobbe",
    tfds_version="0.0.1",
    action_bounds=(-5.0, 5.0),
)

LVIS_CONFIG = DatasetConfig(
    name="lvis",
    tfds_version="1.0.0",
)

BRIDGE_CONFIG = DatasetConfig(
    name="bridge_v2_oxe",
    original_image_size=(256, 256),
    frame_offset=0,
)

GNM_CONFIG = DatasetConfig(
    name="gnm",
    is_navigation=True,
    force_no_wrist_image=True,
)

LIBERO_CONFIG = DatasetConfig(
    name="libero",
    original_image_size=(256, 256),
)

MOLMOACT_CONFIG = DatasetConfig(
    name="molmoact_dataset",
    original_image_size=(224, 224),
)

ALOHA_CONFIG = DatasetConfig(
    name="aloha",
    is_bimanual=True,
    needs_wrist_rotation=True,
)

# Datasets requiring wrist rotation (consolidated list)
WRIST_ROTATION_PATTERNS: list[str] = [
    "droid",
    "aloha",
    "mobile_aloha",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "berkeley_fanuc_manipulation",
    "berkeley_autolab_ur5",
    "fmb",
]

# Pre-register configs for datasets that need TFDS version pinning
# This ensures get_tfds_name_with_version() works even before dataset classes are loaded
_PRE_REGISTERED_CONFIGS: list[DatasetConfig] = [
    DROID_CONFIG,
    FMB_CONFIG,
    DOBBE_CONFIG,
    LVIS_CONFIG,
    BRIDGE_CONFIG,
    GNM_CONFIG,
    LIBERO_CONFIG,
    MOLMOACT_CONFIG,
    ALOHA_CONFIG,
]


# =============================================================================
# DATASET METADATA
# =============================================================================


@dataclass
class DatasetMetadata:
    """Metadata about a registered dataset.

    Attributes:
        cls: The dataset class.
        name: Registered name (may be None for pattern-only registrations).
        config: Dataset configuration.
        is_vqa: Whether this is a VQA dataset.
        vqa_dataset_id: Auto-assigned VQA dataset ID (None for non-VQA).
        requires_hash_tables: Whether the dataset needs pre-built hash tables.
        priority: Priority for pattern matching (higher = checked first).
        has_matcher: Whether this dataset has a pattern matcher.
    """

    cls: type[BaseRobotDataset]
    name: str | None = None
    config: DatasetConfig = field(default_factory=DatasetConfig)
    is_vqa: bool = False
    vqa_dataset_id: int | None = None
    requires_hash_tables: bool = False
    priority: int = 0
    has_matcher: bool = False


# =============================================================================
# REGISTRIES
# =============================================================================

# Registry for exact name matches with metadata
_DATASET_METADATA: dict[str, DatasetMetadata] = {}

# Registry for exact name matches
DATASET_REGISTRY: dict[str, type[BaseRobotDataset]] = {}

# Registry for pattern-based matches (checked in order)
DATASET_MATCHERS: list[tuple[Callable[[str], bool], type[BaseRobotDataset], int]] = []

# Configuration registry (name -> config)
_DATASET_CONFIGS: dict[str, DatasetConfig] = {}

# Populate with pre-registered configs (for TFDS version pinning)
for _cfg in _PRE_REGISTERED_CONFIGS:
    if _cfg.name:
        _DATASET_CONFIGS[_cfg.name] = _cfg

# Pattern-based configs (checked in order)
_PATTERN_CONFIGS: list[tuple[str, DatasetConfig]] = [
    ("gnm_", GNM_CONFIG),
    ("libero", LIBERO_CONFIG),
    ("aloha", ALOHA_CONFIG),
    ("mobile_aloha", ALOHA_CONFIG),
]

# VQA dataset names registry
VQA_DATASET_NAMES: set[str] = set()

# VQA dataset ID mappings (auto-assigned)
_VQA_DATASET_ID_MAP: dict[str, int] = {}
_VQA_DATASET_ID_TO_NAME: dict[int, str] = {}
_NEXT_VQA_ID: int = 1  # Start from 1, 0 is reserved for non-VQA


# =============================================================================
# REGISTRATION DECORATOR
# =============================================================================


def register_dataset(
    name: str | None = None,
    matcher: Callable[[str], bool] | None = None,
    is_vqa: bool = False,
    priority: int = 0,
    requires_hash_tables: bool = False,
    config: DatasetConfig | None = None,
):
    """Decorator to register a dataset class with optional configuration.

    Args:
        name: Exact name to register (e.g., "droid").
        matcher: A callable that takes a dataset name and returns True if this class handles it.
        is_vqa: If True, registers this as a VQA dataset (auto-assigns ID).
        priority: Higher priority matchers are checked first. Default is 0.
        requires_hash_tables: If True, indicates this dataset needs pre-built hash tables.
        config: Optional DatasetConfig with dataset-specific settings.

    Returns:
        The decorator function that registers the class.

    Example:
        @register_dataset(
            name="droid",
            config=DatasetConfig(needs_wrist_rotation=True),
            requires_hash_tables=True,
        )
        class DroidDataset(BaseRobotDataset):
            ...
    """
    global _NEXT_VQA_ID

    def decorator(cls: type[BaseRobotDataset]) -> type[BaseRobotDataset]:
        global _NEXT_VQA_ID

        vqa_dataset_id = None

        # Create or use provided config
        dataset_config = config if config is not None else DatasetConfig(name=name or "")
        if name and not dataset_config.name:
            dataset_config.name = name

        if name is not None:
            DATASET_REGISTRY[name] = cls
            _DATASET_CONFIGS[name] = dataset_config

            if is_vqa:
                VQA_DATASET_NAMES.add(name)
                vqa_dataset_id = _NEXT_VQA_ID
                _VQA_DATASET_ID_MAP[name] = vqa_dataset_id
                _VQA_DATASET_ID_TO_NAME[vqa_dataset_id] = name
                _NEXT_VQA_ID += 1

            # Store rich metadata
            _DATASET_METADATA[name] = DatasetMetadata(
                cls=cls,
                name=name,
                config=dataset_config,
                is_vqa=is_vqa,
                vqa_dataset_id=vqa_dataset_id,
                requires_hash_tables=requires_hash_tables,
                priority=priority,
                has_matcher=matcher is not None,
            )

        if matcher is not None:
            entry = (matcher, cls, priority)
            inserted = False
            for i, (_, _, p) in enumerate(DATASET_MATCHERS):
                if priority > p:
                    DATASET_MATCHERS.insert(i, entry)
                    inserted = True
                    break
            if not inserted:
                DATASET_MATCHERS.append(entry)

        # Store registration info on class for introspection
        if not hasattr(cls, "_registry_info"):
            cls._registry_info = {}
        cls._registry_info["name"] = name
        cls._registry_info["is_vqa"] = is_vqa
        cls._registry_info["vqa_dataset_id"] = vqa_dataset_id
        cls._registry_info["has_matcher"] = matcher is not None
        cls._registry_info["requires_hash_tables"] = requires_hash_tables
        cls._registry_info["config"] = dataset_config

        if requires_hash_tables:
            cls.REQUIRES_HASH_TABLES = True

        return cls

    return decorator


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================


def get_dataset_class(dataset_name: str) -> type[BaseRobotDataset] | None:
    """Get the dataset class for a given dataset name.

    First checks exact name matches in DATASET_REGISTRY, then
    falls back to pattern matching via DATASET_MATCHERS.
    """
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]

    for matcher, cls, _ in DATASET_MATCHERS:
        if matcher(dataset_name):
            return cls

    return None


def get_dataset_config(dataset_name: str) -> DatasetConfig | None:
    """Get configuration for a dataset by name.

    First checks exact matches, then pattern matches.
    """
    if dataset_name in _DATASET_CONFIGS:
        return _DATASET_CONFIGS[dataset_name]

    for pattern, config in _PATTERN_CONFIGS:
        if pattern in dataset_name:
            return config

    return None


def register_dataset_config(name: str, config: DatasetConfig) -> None:
    """Register a new dataset configuration."""
    _DATASET_CONFIGS[name] = config


def get_tfds_name_with_version(dataset_name: str) -> str:
    """Get the TFDS dataset name with version if configured."""
    config = get_dataset_config(dataset_name)
    if config and config.tfds_version:
        return f"{dataset_name}:{config.tfds_version}"
    return dataset_name


def get_action_bounds(dataset_name: str) -> tuple[float, float] | None:
    """Get action bounds for filtering if configured."""
    config = get_dataset_config(dataset_name)
    return config.action_bounds if config else None


def needs_wrist_rotation(dataset_name: str) -> bool:
    """Check if a dataset requires wrist camera rotation."""
    config = get_dataset_config(dataset_name)
    if config and config.needs_wrist_rotation:
        return True
    return any(pattern in dataset_name for pattern in WRIST_ROTATION_PATTERNS)


def is_navigation_dataset(dataset_name: str) -> bool:
    """Check if a dataset is a navigation dataset."""
    config = get_dataset_config(dataset_name)
    if config:
        return config.is_navigation
    return "gnm_" in dataset_name


def is_bimanual_dataset(dataset_name: str) -> bool:
    """Check if a dataset is bimanual."""
    config = get_dataset_config(dataset_name)
    if config:
        return config.is_bimanual
    return "aloha" in dataset_name.lower()


def is_vqa_dataset(dataset_name: str) -> bool:
    """Check if a dataset is a VQA dataset."""
    return dataset_name in VQA_DATASET_NAMES


def get_vqa_dataset_id(dataset_name: str) -> int:
    """Get the auto-assigned VQA dataset ID for a dataset.

    VQA dataset IDs are automatically assigned when datasets are registered
    with is_vqa=True. ID 0 is reserved for non-VQA samples.
    """
    return _VQA_DATASET_ID_MAP.get(dataset_name, 0)


def get_vqa_dataset_name(dataset_id: int) -> str | None:
    """Get the dataset name for a VQA dataset ID."""
    return _VQA_DATASET_ID_TO_NAME.get(dataset_id)


def get_num_vqa_datasets() -> int:
    """Get the total number of registered VQA datasets."""
    return len(_VQA_DATASET_ID_MAP)


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata | None:
    """Get full metadata for a registered dataset."""
    return _DATASET_METADATA.get(dataset_name)


def requires_hash_tables(dataset_name: str) -> bool:
    """Check if a dataset requires pre-built hash tables."""
    metadata = _DATASET_METADATA.get(dataset_name)
    if metadata:
        return metadata.requires_hash_tables

    cls = get_dataset_class(dataset_name)
    if cls and hasattr(cls, "REQUIRES_HASH_TABLES"):
        return cls.REQUIRES_HASH_TABLES

    return False


def list_registered_datasets() -> dict[str, list]:
    """List all registered datasets for debugging."""
    return {
        "exact": list(DATASET_REGISTRY.keys()),
        "patterns": [f"{cls.__name__} (priority={priority})" for _, cls, priority in DATASET_MATCHERS],
        "vqa": list(VQA_DATASET_NAMES),
        "vqa_ids": dict(_VQA_DATASET_ID_MAP),
        "configs": {name: config.name for name, config in _DATASET_CONFIGS.items()},
    }


def get_dataset_class_with_fallback(
    dataset_name: str,
    fallback_cls: type[BaseRobotDataset],
) -> type[BaseRobotDataset]:
    """Get dataset class with a fallback if not found."""
    cls = get_dataset_class(dataset_name)
    return cls if cls is not None else fallback_cls


# Public aliases
VQA_DATASET_ID_MAP = _VQA_DATASET_ID_MAP
VQA_DATASET_ID_TO_NAME = _VQA_DATASET_ID_TO_NAME
