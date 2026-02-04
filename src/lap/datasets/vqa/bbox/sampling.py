"""Object sampling and formatting utilities for VQA datasets.

This module provides functions for sampling objects from bbox annotations
and formatting them into prompts and captions.
"""

import random
import json

import tensorflow as tf

from lap.datasets.vqa.bbox.coord_utils import bbox_to_loc_tokens


def sample_and_format_objects(
    objects_json: bytes,
    max_objects: int = 2,
    seed: int | None = None,
) -> tuple[bytes, bytes]:
    """Sample objects and format them into prompt labels and caption.

    This function is meant to be called via tf.py_function during iteration.

    Args:
        objects_json: JSON-serialized list of objects with 'label' and 'bbox'.
        max_objects: Maximum number of objects to include (randomly sampled if more).
        seed: Random seed for sampling (if None, uses true randomness).

    Returns:
        Tuple of (prompt_labels, caption) as bytes.
    """
    if not objects_json:
        return b"", b""

    try:
        objects = json.loads(objects_json.decode("utf-8"))
        if not objects:
            return b"", b""

        # Randomly sample max_objects if there are more
        if len(objects) > max_objects:
            if seed is not None:
                rng = random.Random(seed)
                objects = rng.sample(objects, max_objects)
            else:
                objects = random.sample(objects, max_objects)

        # Build prompt with all object labels
        labels = [obj["label"] for obj in objects]
        unique_labels = list(dict.fromkeys(labels))  # Preserve order, remove duplicates
        prompt_labels = ", ".join(unique_labels)

        # Build caption with all bboxes (already letterbox-transformed)
        caption_parts = []
        for obj in objects:
            label = obj["label"]
            bbox = obj["bbox"]  # [x_min, y_min, x_max, y_max] already transformed
            loc_str = bbox_to_loc_tokens(bbox[0], bbox[1], bbox[2], bbox[3])
            caption_parts.append(f"{loc_str} {label}")

        caption = " ; ".join(caption_parts)

        return prompt_labels.encode("utf-8"), caption.encode("utf-8")

    except Exception:
        return b"", b""


def sample_and_format_objects_tf(
    objects_data: tf.Tensor,
    max_objects: int = 2,
    seed_pair: tuple[int, tf.Tensor] | None = None,
    direction_prob: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Sample objects and format them into prompt labels and caption (pure TensorFlow).

    This function uses pure TensorFlow operations, avoiding py_function overhead.
    The input format is a pipe-delimited string: "label1|loc_tokens1|direction1;label2|loc_tokens2|direction2;..."
    where loc_tokens is already formatted as "<loc...><loc...><loc...><loc...>"
    and direction is like "move forward", "move left", etc.

    With probability direction_prob, the caption will use direction instead of loc_tokens.
    When using direction caption, only ONE object is sampled (direction refers to single object).
    When using bbox caption, up to max_objects are sampled.

    Args:
        objects_data: tf.string tensor with pipe-delimited objects data.
        max_objects: Maximum number of objects to include for bbox caption (randomly sampled if more).
        seed_pair: Tuple of (base_seed, hash_value) for stateless random sampling.
        direction_prob: Probability of using direction caption instead of bbox (default 0.0).

    Returns:
        Tuple of (prompt_labels, caption) as tf.string tensors.
    """
    # Handle empty input
    is_empty = tf.equal(tf.strings.length(objects_data), 0)

    def empty_result():
        return tf.constant(""), tf.constant("")

    def process_objects():
        # Split by semicolon to get individual objects
        objects = tf.strings.split(objects_data, ";")
        num_objects = tf.shape(objects)[0]

        # First, decide whether to use direction or bbox caption
        if seed_pair is not None:
            dir_seed = (seed_pair[0] + 7919, seed_pair[1])
            use_direction = tf.random.stateless_uniform([], seed=dir_seed, dtype=tf.float32) < direction_prob
        else:
            use_direction = tf.random.uniform([], dtype=tf.float32) < direction_prob

        # Determine how many objects to sample based on caption type
        # Direction caption: always 1 object
        # Bbox caption: up to max_objects
        effective_max = tf.cond(use_direction, lambda: 1, lambda: max_objects)

        # Sample indices
        def sample_indices():
            if seed_pair is not None:
                random_vals = tf.random.stateless_uniform(
                    [num_objects], seed=seed_pair, dtype=tf.float32
                )
            else:
                random_vals = tf.random.uniform([num_objects], dtype=tf.float32)
            shuffled = tf.argsort(random_vals)
            return shuffled[:effective_max]

        def all_indices():
            return tf.range(tf.minimum(num_objects, effective_max))

        selected_indices = tf.cond(
            num_objects > effective_max,
            sample_indices,
            all_indices,
        )

        # Gather selected objects
        selected_objects = tf.gather(objects, selected_indices)

        # Split each object into label, loc_tokens, and direction
        def split_object(obj):
            parts = tf.strings.split(obj, "|")
            label = parts[0]
            loc_tokens = parts[1]
            # Handle both old format (2 parts) and new format (3 parts)
            direction = tf.cond(
                tf.greater_equal(tf.shape(parts)[0], 3),
                lambda: parts[2],
                lambda: tf.constant(""),
            )
            return label, loc_tokens, direction

        labels_locs_dirs = tf.map_fn(
            split_object,
            selected_objects,
            fn_output_signature=(
                tf.TensorSpec([], tf.string),
                tf.TensorSpec([], tf.string),
                tf.TensorSpec([], tf.string),
            ),
        )
        labels = labels_locs_dirs[0]
        loc_tokens = labels_locs_dirs[1]
        directions = labels_locs_dirs[2]

        # Check if we have valid direction data (non-empty)
        has_direction = tf.greater(tf.strings.length(directions[0]), 0)
        use_direction_final = tf.logical_and(use_direction, has_direction)

        def bbox_result():
            # Build prompt_labels from all sampled labels
            prompt_labels = tf.strings.reduce_join(labels, separator=", ")
            # Build caption: "loc_tokens label" joined by " ; "
            caption_parts = tf.strings.join([loc_tokens, labels], separator=" ")
            caption = tf.strings.reduce_join(caption_parts, separator=" ; ")
            return prompt_labels, caption

        def direction_result():
            # For direction, use only the first (and only) sampled object
            prompt_labels = labels[0]
            caption = directions[0]
            return prompt_labels, caption

        return tf.cond(use_direction_final, direction_result, bbox_result)

    return tf.cond(is_empty, empty_result, process_objects)


def sample_and_format_objects_direction_tf(
    objects_data: tf.Tensor,
    seed_pair: tuple[int, tf.Tensor] | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Sample ONE object and format for direction-based VQA (pure TensorFlow).

    This function uses pure TensorFlow operations for direction-based output.
    The input format is a pipe-delimited string: "label1|direction1;label2|direction2;..."

    Direction caption refers to a single object, so we always sample exactly 1 object.

    Args:
        objects_data: tf.string tensor with pipe-delimited objects data.
        seed_pair: Tuple of (base_seed, hash_value) for stateless random sampling.

    Returns:
        Tuple of (prompt_label, direction_caption) as tf.string tensors.
        - prompt_label: single object label for the prompt.
        - direction_caption: the direction string for the caption (e.g., "move left").
    """
    # Handle empty input
    is_empty = tf.equal(tf.strings.length(objects_data), 0)

    def empty_result():
        return tf.constant(""), tf.constant("")

    def process_objects():
        # Split by semicolon to get individual objects
        objects = tf.strings.split(objects_data, ";")
        num_objects = tf.shape(objects)[0]

        # Always sample exactly 1 object
        def sample_one_index():
            if seed_pair is not None:
                random_vals = tf.random.stateless_uniform(
                    [num_objects], seed=seed_pair, dtype=tf.float32
                )
            else:
                random_vals = tf.random.uniform([num_objects], dtype=tf.float32)
            # Get index of minimum random value (random selection of 1)
            return tf.argmin(random_vals, output_type=tf.int32)

        selected_idx = sample_one_index()

        # Get the selected object
        selected_object = objects[selected_idx]

        # Split into label and direction
        parts = tf.strings.split(selected_object, "|")
        label = parts[0]
        direction = parts[1]

        return label, direction

    return tf.cond(is_empty, empty_result, process_objects)
