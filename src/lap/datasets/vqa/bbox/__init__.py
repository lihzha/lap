"""Bounding box VQA utilities package.

This package provides utilities for bounding box VQA datasets, organized
into focused modules:

- prompts: Prompt templates for bbox detection and direction tasks
- coord_utils: Bbox coordinate conversion and letterbox transformations
- direction: Direction classification utilities
- key_extractors: Episode key extraction for bbox annotation lookup
- sampling: Object sampling and formatting utilities
- table_builder: Lookup table construction for efficient annotation access

Commonly used functions and constants are re-exported from this __init__.py.
"""

# Prompt templates
from lap.datasets.vqa.bbox.prompts import (
    DIRECTION_PROMPT_PARTS,
    GENERAL_BBOX_PROMPT_PARTS,
    ROBOT_BBOX_PROMPT_PARTS,
    ROBOT_BBOX_PROMPT_PARTS_EE,
    ROBOT_BBOX_PROMPT_PARTS_OXE,
    ROBOT_DIRECTION_PROMPT_PARTS_EE,
    ROBOT_DIRECTION_PROMPT_PARTS_OXE,
    sample_prompt_tf,
)

# Coordinate utilities
from lap.datasets.vqa.bbox.coord_utils import (
    bbox_coords_to_text_tf,
    bbox_to_loc_tokens,
    bbox_to_text_tf,
    format_bbox_caption,
    rotate_bbox_loc_tokens_180_tf,
    transform_bbox_for_letterbox,
)

# Direction utilities
from lap.datasets.vqa.bbox.direction import (
    compute_direction_from_bbox,
    direction_from_bbox_tf,
    rotate_direction_180_tf,
)

# Key extractors
from lap.datasets.vqa.bbox.key_extractors import (
    bridge_key_extractor,
    droid_key_extractor,
    oxe_key_extractor,
)

# Sampling utilities
from lap.datasets.vqa.bbox.sampling import (
    sample_and_format_objects,
    sample_and_format_objects_direction_tf,
    sample_and_format_objects_tf,
)

# Table builders
from lap.datasets.vqa.bbox.table_builder import (
    build_annotated_keys_and_frame_table_v2,
    build_annotated_keys_set,
    build_frame_objects_table_v2,
    build_frame_objects_table_v2_direction,
    count_annotated_frames,
)

__all__ = [
    # Prompts
    "DIRECTION_PROMPT_PARTS",
    "GENERAL_BBOX_PROMPT_PARTS",
    "ROBOT_BBOX_PROMPT_PARTS",
    "ROBOT_BBOX_PROMPT_PARTS_EE",
    "ROBOT_BBOX_PROMPT_PARTS_OXE",
    "ROBOT_DIRECTION_PROMPT_PARTS_EE",
    "ROBOT_DIRECTION_PROMPT_PARTS_OXE",
    "sample_prompt_tf",
    # Coordinate utilities
    "bbox_coords_to_text_tf",
    "bbox_to_loc_tokens",
    "bbox_to_text_tf",
    "format_bbox_caption",
    "rotate_bbox_loc_tokens_180_tf",
    "transform_bbox_for_letterbox",
    # Direction utilities
    "compute_direction_from_bbox",
    "direction_from_bbox_tf",
    "rotate_direction_180_tf",
    # Key extractors
    "bridge_key_extractor",
    "droid_key_extractor",
    "oxe_key_extractor",
    # Sampling utilities
    "sample_and_format_objects",
    "sample_and_format_objects_direction_tf",
    "sample_and_format_objects_tf",
    # Table builders
    "build_annotated_keys_and_frame_table_v2",
    "build_annotated_keys_set",
    "build_frame_objects_table_v2",
    "build_frame_objects_table_v2_direction",
    "count_annotated_frames",
]
