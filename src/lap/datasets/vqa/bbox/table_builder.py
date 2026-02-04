"""Lookup table builders for bbox VQA datasets.

This module provides functions for building TensorFlow lookup tables
that map frame identifiers to object annotations for efficient access
during dataset iteration.
"""

import json
import logging
import os
from typing import Callable

import tensorflow as tf

from lap.datasets.vqa.bbox.coord_utils import (
    bbox_to_loc_tokens,
    transform_bbox_for_letterbox,
)
from lap.datasets.vqa.bbox.direction import compute_direction_from_bbox


def build_frame_objects_table_v2(
    bbox_annotations_dir: str,
    key_extractor: Callable[[dict], str | None],
    dataset_name: str = "",
    orig_size: tuple[int, int] = (256, 256),
    target_size: tuple[int, int] = (224, 224),
    target_only: bool = False,
    direction_slope: float = 2.0,
) -> tf.lookup.StaticHashTable:
    """Build a lookup table from key--frame_idx to pipe-delimited objects string.

    This version stores data in a TF-parseable format instead of JSON, enabling
    pure TensorFlow operations for sampling and formatting.

    Format: "label1|<loc...>|direction1;label2|<loc...>|direction2;..."
    where direction is "move forward", "move left", etc.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files.
        key_extractor: Function that takes episode_data dict and returns the key string.
        dataset_name: Optional dataset name for logging.
        orig_size: Original image (width, height) for letterbox transformation.
        target_size: Target image (width, height) for letterbox transformation.
        target_only: If True, only include objects where is_target is True.
        direction_slope: Slope parameter for direction computation (default 2.0).

    Returns:
        tf.lookup.StaticHashTable mapping "key--frame_idx" to pipe-delimited objects string.
    """
    log_prefix = f" for {dataset_name}" if dataset_name else ""
    logging.info(f"Building frame objects lookup table (v2){log_prefix}...")

    orig_w, orig_h = orig_size
    target_w, target_h = target_size

    frame_to_objects = {}
    
    # Counters for filtered bboxes
    total_bboxes = 0
    invalid_bbox_count = 0
    missing_label_count = 0
    skipped_non_target = 0

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        if "merged" in jsonl_file:
            continue
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Use the provided key extractor to get the lookup key
                episode_key = key_extractor(episode_data)
                if not episode_key:
                    continue

                labels = episode_data.get("labels", [])
                for label_entry in labels:
                    frame_idx = label_entry.get("frame")
                    all_objects = label_entry.get("all_objects", [])

                    if frame_idx is None or not all_objects:
                        continue

                    key = f"{episode_key}--{frame_idx}"

                    objects_list = []
                    for obj in all_objects:
                        total_bboxes += 1
                        obj_label = obj.get("label", "")
                        bbox = obj.get("bbox", [])
                        is_target = obj.get("is_target", False)

                        if not obj_label:
                            missing_label_count += 1
                            continue
                        if len(bbox) != 4:
                            invalid_bbox_count += 1
                            continue

                        # Skip non-target objects if target_only is enabled
                        if target_only and not is_target:
                            skipped_non_target += 1
                            continue

                        # Normalize bbox (bbox values are in 0-1000 range in JSONL)
                        y_min_raw = max(0.0, min(1.0, float(bbox[0]) / 1000.0))
                        x_min_raw = max(0.0, min(1.0, float(bbox[1]) / 1000.0))
                        y_max_raw = max(0.0, min(1.0, float(bbox[2]) / 1000.0))
                        x_max_raw = max(0.0, min(1.0, float(bbox[3]) / 1000.0))

                        # Pre-apply letterbox transformation
                        x_min, y_min, x_max, y_max = transform_bbox_for_letterbox(
                            x_min_raw, y_min_raw, x_max_raw, y_max_raw,
                            orig_w, orig_h, target_w, target_h,
                        )

                        # Pre-compute loc tokens
                        loc_tokens = bbox_to_loc_tokens(x_min, y_min, x_max, y_max)

                        # Pre-compute direction with "move " prefix
                        direction = compute_direction_from_bbox(
                            x_min, y_min, x_max, y_max,
                            slope=direction_slope,
                            add_move_prefix=True,
                        )

                        # Store as "label|loc_tokens|direction"
                        objects_list.append(f"{obj_label}|{loc_tokens}|{direction}")

                    if objects_list:
                        if key in frame_to_objects:
                            frame_to_objects[key].extend(objects_list)
                        else:
                            frame_to_objects[key] = objects_list

    # Convert to lookup table with semicolon-delimited values
    keys = []
    values = []
    for k, v in frame_to_objects.items():
        keys.append(k)
        values.append(";".join(v))

    logging.info(f"Built frame objects table (v2) with {len(keys)} entries{log_prefix}")
    logging.info(f"  Total bboxes processed: {total_bboxes}{log_prefix}")
    if invalid_bbox_count > 0:
        logging.warning(f"  Filtered {invalid_bbox_count} bboxes with invalid coordinate count (not 4){log_prefix}")
    if missing_label_count > 0:
        logging.warning(f"  Filtered {missing_label_count} bboxes with missing labels{log_prefix}")
    if skipped_non_target > 0:
        logging.info(f"  Skipped {skipped_non_target} non-target bboxes (target_only={target_only}){log_prefix}")
    
    # Debug: Log sample keys to help diagnose mismatches
    if keys:
        sample_keys = keys[:5]
        logging.info(f"Sample JSONL lookup keys{log_prefix}: {sample_keys}")

    if not keys:
        # Return table with dummy entry (TF doesn't allow empty tables)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["__dummy_key__"], dtype=tf.string),
                tf.constant([""], dtype=tf.string),
            ),
            default_value=tf.constant("", dtype=tf.string),
        )

    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(values, dtype=tf.string),
        ),
        default_value=tf.constant("", dtype=tf.string),
    )


def build_frame_objects_table_v2_direction(
    bbox_annotations_dir: str,
    key_extractor: Callable[[dict], str | None],
    dataset_name: str = "",
    orig_size: tuple[int, int] = (256, 256),
    target_size: tuple[int, int] = (224, 224),
    direction_slope: float = 2.0,
) -> tf.lookup.StaticHashTable:
    """Build a lookup table from key--frame_idx to pipe-delimited objects with directions.

    This version computes direction strings from bbox coordinates instead of loc tokens.

    Format: "label1|direction1;label2|direction2;..."

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files.
        key_extractor: Function that takes episode_data dict and returns the key string.
        dataset_name: Optional dataset name for logging.
        orig_size: Original image (width, height) for letterbox transformation.
        target_size: Target image (width, height) for letterbox transformation.
        direction_slope: Slope parameter for direction boundaries.

    Returns:
        tf.lookup.StaticHashTable mapping "key--frame_idx" to pipe-delimited objects string.
    """
    log_prefix = f" for {dataset_name}" if dataset_name else ""
    logging.info(f"Building frame objects direction lookup table{log_prefix}...")

    orig_w, orig_h = orig_size
    target_w, target_h = target_size

    frame_to_objects = {}
    
    # Counters for filtered bboxes
    total_bboxes = 0
    invalid_bbox_count = 0
    missing_label_count = 0

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        if "merged" in jsonl_file:
            continue
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Use the provided key extractor to get the lookup key
                episode_key = key_extractor(episode_data)
                if not episode_key:
                    continue

                labels = episode_data.get("labels", [])
                for label_entry in labels:
                    frame_idx = label_entry.get("frame")
                    all_objects = label_entry.get("all_objects", [])

                    if frame_idx is None or not all_objects:
                        continue

                    key = f"{episode_key}--{frame_idx}"

                    objects_list = []
                    for obj in all_objects:
                        total_bboxes += 1
                        obj_label = obj.get("label", "")
                        bbox = obj.get("bbox", [])

                        if not obj_label:
                            missing_label_count += 1
                            continue
                        if len(bbox) != 4:
                            invalid_bbox_count += 1
                            continue

                        # Normalize bbox (bbox values are in 0-1000 range in JSONL)
                        y_min_raw = max(0.0, min(1.0, float(bbox[0]) / 1000.0))
                        x_min_raw = max(0.0, min(1.0, float(bbox[1]) / 1000.0))
                        y_max_raw = max(0.0, min(1.0, float(bbox[2]) / 1000.0))
                        x_max_raw = max(0.0, min(1.0, float(bbox[3]) / 1000.0))

                        # Pre-apply letterbox transformation
                        x_min, y_min, x_max, y_max = transform_bbox_for_letterbox(
                            x_min_raw, y_min_raw, x_max_raw, y_max_raw,
                            orig_w, orig_h, target_w, target_h,
                        )

                        # Compute direction from bbox
                        direction = compute_direction_from_bbox(
                            x_min, y_min, x_max, y_max, slope=direction_slope
                        )

                        # Store as "label|direction"
                        objects_list.append(f"{obj_label}|{direction}")

                    if objects_list:
                        if key in frame_to_objects:
                            frame_to_objects[key].extend(objects_list)
                        else:
                            frame_to_objects[key] = objects_list

    # Convert to lookup table with semicolon-delimited values
    keys = []
    values = []
    for k, v in frame_to_objects.items():
        keys.append(k)
        values.append(";".join(v))

    logging.info(f"Built frame objects direction table with {len(keys)} entries{log_prefix}")
    logging.info(f"  Total bboxes processed: {total_bboxes}{log_prefix}")
    if invalid_bbox_count > 0:
        logging.warning(f"  Filtered {invalid_bbox_count} bboxes with invalid coordinate count (not 4){log_prefix}")
    if missing_label_count > 0:
        logging.warning(f"  Filtered {missing_label_count} bboxes with missing labels{log_prefix}")

    if not keys:
        # Return table with dummy entry (TF doesn't allow empty tables)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["__dummy_key__"], dtype=tf.string),
                tf.constant([""], dtype=tf.string),
            ),
            default_value=tf.constant("", dtype=tf.string),
        )

    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(values, dtype=tf.string),
        ),
        default_value=tf.constant("", dtype=tf.string),
    )


def build_annotated_keys_set(
    bbox_annotations_dir: str,
    key_extractor: Callable[[dict], str | None],
) -> set[str]:
    """Build a set of keys (uuids or episode_paths) that have bbox annotations.

    This is used for trajectory-level filtering to skip entire trajectories
    that have no annotated frames, which significantly speeds up iteration.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files.
        key_extractor: Function that takes episode_data dict and returns the key string.

    Returns:
        Set of keys that have at least one annotated frame.
    """
    annotated_keys = set()

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        if "merged" in jsonl_file:
            continue
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                episode_key = key_extractor(episode_data)
                if episode_key:
                    annotated_keys.add(episode_key)

    return annotated_keys


def count_annotated_frames(
    bbox_annotations_dir: str,
    key_extractor: Callable[[dict], str | None],
) -> int:
    """Count total number of annotated frames in JSONL files.

    This is used to compute the number of transitions for dataset statistics.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files.
        key_extractor: Function that takes episode_data dict and returns the key string.

    Returns:
        Total count of annotated frames across all JSONL files.
    """
    total_frames = 0

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        if "merged" in jsonl_file:
            continue
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                episode_key = key_extractor(episode_data)
                if not episode_key:
                    continue

                labels = episode_data.get("labels", [])
                for label_entry in labels:
                    frame_idx = label_entry.get("frame")
                    all_objects = label_entry.get("all_objects", [])

                    # Only count frames that have valid annotations
                    if frame_idx is not None and all_objects:
                        total_frames += 1

    return total_frames


def build_annotated_keys_and_frame_table_v2(
    bbox_annotations_dir: str,
    key_extractor: Callable[[dict], str | None],
    dataset_name: str = "",
    orig_size: tuple[int, int] = (256, 256),
    target_size: tuple[int, int] = (224, 224),
    target_only: bool = False,
    direction_slope: float = 2.0,
) -> tuple[set[str], tf.lookup.StaticHashTable, int]:
    """Build both annotated keys set and frame objects table in a single pass.

    This optimized version scans JSONL files only once to build both:
    1. A set of episode keys that have annotations (for trajectory filtering)
    2. A lookup table mapping "key--frame_idx" to pipe-delimited objects string
    3. A count of total annotated frames (to avoid another scan)

    This is significantly faster than calling build_annotated_keys_set(),
    build_frame_objects_table_v2(), and count_annotated_frames() separately,
    especially for large datasets like Bridge.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files.
        key_extractor: Function that takes episode_data dict and returns the key string.
        dataset_name: Optional dataset name for logging.
        orig_size: Original image (width, height) for letterbox transformation.
        target_size: Target image (width, height) for letterbox transformation.
        target_only: If True, only include objects where is_target is True.
        direction_slope: Slope parameter for direction computation (default 2.0).

    Returns:
        Tuple of (annotated_keys_set, frame_objects_lookup_table, num_annotated_frames).
    """
    log_prefix = f" for {dataset_name}" if dataset_name else ""
    logging.info(f"Building annotated keys set and frame objects table (v2) in single pass{log_prefix}...")

    orig_w, orig_h = orig_size
    target_w, target_h = target_size

    annotated_keys = set()
    frame_to_objects = {}
    
    # Counters for filtered bboxes
    total_bboxes = 0
    invalid_bbox_count = 0
    missing_label_count = 0
    skipped_non_target = 0
    num_annotated_frames = 0  # Count frames with valid annotations

    # Always use merged file - simpler and faster for sequential processing
    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))
    merged_files = [f for f in jsonl_files if "merged" in f.lower()]
    
    if not merged_files:
        raise FileNotFoundError(
            f"No merged JSONL file found in {bbox_annotations_dir}. "
            f"Please create a merged file (e.g., 'bridge_bbox_merged.jsonl') containing all annotations."
        )
    
    # Use first merged file if multiple exist
    jsonl_file = merged_files[0]
    if len(merged_files) > 1:
        logging.warning(f"Multiple merged files found, using: {jsonl_file}{log_prefix}")
    else:
        logging.info(f"Using merged JSONL file: {jsonl_file}{log_prefix}")
    
    # Process merged file sequentially (simpler and faster than parallel processing)
    logging.info(f"Processing merged JSONL file{log_prefix}...")
    
    line_count = 0
    with tf.io.gfile.GFile(jsonl_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            
            line_count += 1
            # Log progress every 100k lines
            if line_count % 100000 == 0:
                logging.info(f"  Processed {line_count:,} lines{log_prefix}...")
            
            try:
                episode_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Use the provided key extractor to get the lookup key
            episode_key = key_extractor(episode_data)
            if not episode_key:
                continue

            # Add to annotated keys set (for trajectory filtering)
            annotated_keys.add(episode_key)

            labels = episode_data.get("labels", [])
            for label_entry in labels:
                frame_idx = label_entry.get("frame")
                all_objects = label_entry.get("all_objects", [])

                if frame_idx is None or not all_objects:
                    continue

                # Count this as an annotated frame
                num_annotated_frames += 1

                key = f"{episode_key}--{frame_idx}"

                objects_list = []
                for obj in all_objects:
                    total_bboxes += 1
                    obj_label = obj.get("label", "")
                    bbox = obj.get("bbox", [])
                    is_target = obj.get("is_target", False)

                    if not obj_label:
                        missing_label_count += 1
                        continue
                    if len(bbox) != 4:
                        invalid_bbox_count += 1
                        continue

                    # Skip non-target objects if target_only is enabled
                    if target_only and not is_target:
                        skipped_non_target += 1
                        continue

                    # Normalize bbox (bbox values are in 0-1000 range in JSONL)
                    y_min_raw = max(0.0, min(1.0, float(bbox[0]) / 1000.0))
                    x_min_raw = max(0.0, min(1.0, float(bbox[1]) / 1000.0))
                    y_max_raw = max(0.0, min(1.0, float(bbox[2]) / 1000.0))
                    x_max_raw = max(0.0, min(1.0, float(bbox[3]) / 1000.0))

                    # Pre-apply letterbox transformation
                    x_min, y_min, x_max, y_max = transform_bbox_for_letterbox(
                        x_min_raw, y_min_raw, x_max_raw, y_max_raw,
                        orig_w, orig_h, target_w, target_h,
                    )

                    # Pre-compute loc tokens
                    loc_tokens = bbox_to_loc_tokens(x_min, y_min, x_max, y_max)

                    # Pre-compute direction with "move " prefix
                    direction = compute_direction_from_bbox(
                        x_min, y_min, x_max, y_max,
                        slope=direction_slope,
                        add_move_prefix=True,
                    )

                    # Store as "label|loc_tokens|direction"
                    objects_list.append(f"{obj_label}|{loc_tokens}|{direction}")

                if objects_list:
                    if key in frame_to_objects:
                        frame_to_objects[key].extend(objects_list)
                    else:
                        frame_to_objects[key] = objects_list
    
    logging.info(f"Finished processing {line_count:,} lines from merged file{log_prefix}")

    # Convert to lookup table with semicolon-delimited values
    keys = []
    values = []
    for k, v in frame_to_objects.items():
        keys.append(k)
        values.append(";".join(v))

    logging.info(f"Built annotated keys set ({len(annotated_keys)} keys) and frame objects table ({len(keys)} entries){log_prefix}")
    logging.info(f"  Total annotated frames: {num_annotated_frames}{log_prefix}")
    logging.info(f"  Total bboxes processed: {total_bboxes}{log_prefix}")
    if invalid_bbox_count > 0:
        logging.warning(f"  Filtered {invalid_bbox_count} bboxes with invalid coordinate count (not 4){log_prefix}")
    if missing_label_count > 0:
        logging.warning(f"  Filtered {missing_label_count} bboxes with missing labels{log_prefix}")
    if skipped_non_target > 0:
        logging.info(f"  Skipped {skipped_non_target} non-target bboxes (target_only={target_only}){log_prefix}")
    
    # Debug: Log sample keys to help diagnose mismatches
    if keys:
        sample_keys = keys[:5]
        logging.info(f"Sample JSONL lookup keys{log_prefix}: {sample_keys}")

    if not keys:
        # Return table with dummy entry (TF doesn't allow empty tables)
        lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["__dummy_key__"], dtype=tf.string),
                tf.constant([""], dtype=tf.string),
            ),
            default_value=tf.constant("", dtype=tf.string),
        )
    else:
        lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(keys, dtype=tf.string),
                tf.constant(values, dtype=tf.string),
            ),
            default_value=tf.constant("", dtype=tf.string),
        )

    return annotated_keys, lookup_table, num_annotated_frames
