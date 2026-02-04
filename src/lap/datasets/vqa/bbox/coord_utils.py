"""Bounding box coordinate utilities.

This module provides functions for converting bounding box coordinates
to PaLiGemma loc token format and handling letterbox transformations.
"""

import tensorflow as tf


def bbox_to_loc_tokens(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    num_bins: int = 1024,
) -> str:
    """Convert normalized bbox coordinates to PaLiGemma loc token string.

    Args:
        x_min: Normalized x coordinate of top-left corner (0-1).
        y_min: Normalized y coordinate of top-left corner (0-1).
        x_max: Normalized x coordinate of bottom-right corner (0-1).
        y_max: Normalized y coordinate of bottom-right corner (0-1).
        num_bins: Number of bins for quantization (default 1024).

    Returns:
        Formatted string in PaLiGemma format: "<locYMIN><locXMIN><locYMAX><locXMAX>".
    """
    N = num_bins
    y_min_idx = int(round(y_min * (N - 1)))
    x_min_idx = int(round(x_min * (N - 1)))
    y_max_idx = int(round(y_max * (N - 1)))
    x_max_idx = int(round(x_max * (N - 1)))

    return f"<loc{y_min_idx:04d}><loc{x_min_idx:04d}><loc{y_max_idx:04d}><loc{x_max_idx:04d}>"


def bbox_to_text_tf(bbox: tf.Tensor, num_bins: int = 1024) -> tf.Tensor:
    """Convert bbox tensor to PaLiGemma loc token string (TensorFlow version).

    Args:
        bbox: Tensor of shape [2, 2] with normalized coordinates
              [[x_min, y_min], [x_max, y_max]] where coordinates are in range [0, 1].
        num_bins: Number of bins for quantization (default 1024).

    Returns:
        tf.string tensor with formatted loc tokens: "<locYMIN><locXMIN><locYMAX><locXMAX>".
    """
    top_left = bbox[0]
    bottom_right = bbox[1]

    x_min, y_min = top_left[0], top_left[1]
    x_max, y_max = bottom_right[0], bottom_right[1]

    return bbox_coords_to_text_tf(x_min, y_min, x_max, y_max, num_bins)


def bbox_coords_to_text_tf(
    x_min: tf.Tensor,
    y_min: tf.Tensor,
    x_max: tf.Tensor,
    y_max: tf.Tensor,
    num_bins: int = 1024,
) -> tf.Tensor:
    """Convert bbox coordinate tensors to PaLiGemma loc token string.

    Args:
        x_min: Normalized x coordinate of top-left corner (0-1).
        y_min: Normalized y coordinate of top-left corner (0-1).
        x_max: Normalized x coordinate of bottom-right corner (0-1).
        y_max: Normalized y coordinate of bottom-right corner (0-1).
        num_bins: Number of bins for quantization (default 1024).

    Returns:
        tf.string tensor with formatted loc tokens.
    """
    N = num_bins
    y_min_idx = tf.cast(tf.round(y_min * (N - 1)), tf.int32)
    x_min_idx = tf.cast(tf.round(x_min * (N - 1)), tf.int32)
    y_max_idx = tf.cast(tf.round(y_max * (N - 1)), tf.int32)
    x_max_idx = tf.cast(tf.round(x_max * (N - 1)), tf.int32)

    # Format as loc tokens in PaLiGemma order: y_min, x_min, y_max, x_max
    y_min_token = tf.strings.join(["<loc", tf.strings.as_string(y_min_idx, width=4, fill="0"), ">"])
    x_min_token = tf.strings.join(["<loc", tf.strings.as_string(x_min_idx, width=4, fill="0"), ">"])
    y_max_token = tf.strings.join(["<loc", tf.strings.as_string(y_max_idx, width=4, fill="0"), ">"])
    x_max_token = tf.strings.join(["<loc", tf.strings.as_string(x_max_idx, width=4, fill="0"), ">"])

    return tf.strings.join([y_min_token, x_min_token, y_max_token, x_max_token])


def rotate_bbox_loc_tokens_180_tf(loc_tokens: tf.Tensor, num_bins: int = 1024) -> tf.Tensor:
    """Rotate bbox loc tokens by 180 degrees.
    
    For a 180-degree rotation, bbox coordinates transform as:
    - new_x_min = 1 - x_max
    - new_y_min = 1 - y_max
    - new_x_max = 1 - x_min
    - new_y_max = 1 - y_min
    
    Args:
        loc_tokens: tf.string tensor with loc tokens in format:
                    "<locYMIN><locXMIN><locYMAX><locXMAX>".
        num_bins: Number of bins for quantization (default 1024).
        
    Returns:
        tf.string tensor with rotated loc tokens.
    """
    N = num_bins
    
    def extract_numbers():
        # Use regex to find all <loc####> and extract numbers
        pattern = r"<loc(\d{4})>"
        numbers_str = tf.strings.regex_replace(loc_tokens, pattern, r"\1 ")
        numbers_str = tf.strings.strip(numbers_str)
        numbers = tf.strings.split(numbers_str, " ")
        
        # Parse to integers (should have 4 numbers)
        y_min_idx = tf.strings.to_number(numbers[0], out_type=tf.int32)
        x_min_idx = tf.strings.to_number(numbers[1], out_type=tf.int32)
        y_max_idx = tf.strings.to_number(numbers[2], out_type=tf.int32)
        x_max_idx = tf.strings.to_number(numbers[3], out_type=tf.int32)
        
        # Convert indices to normalized coordinates [0, 1]
        y_min = tf.cast(y_min_idx, tf.float32) / (N - 1)
        x_min = tf.cast(x_min_idx, tf.float32) / (N - 1)
        y_max = tf.cast(y_max_idx, tf.float32) / (N - 1)
        x_max = tf.cast(x_max_idx, tf.float32) / (N - 1)
        
        # Rotate 180 degrees
        new_x_min = 1.0 - x_max
        new_y_min = 1.0 - y_max
        new_x_max = 1.0 - x_min
        new_y_max = 1.0 - y_min
        
        # Convert back to indices
        new_y_min_idx = tf.cast(tf.round(new_y_min * (N - 1)), tf.int32)
        new_x_min_idx = tf.cast(tf.round(new_x_min * (N - 1)), tf.int32)
        new_y_max_idx = tf.cast(tf.round(new_y_max * (N - 1)), tf.int32)
        new_x_max_idx = tf.cast(tf.round(new_x_max * (N - 1)), tf.int32)
        
        # Clamp to valid range
        new_y_min_idx = tf.clip_by_value(new_y_min_idx, 0, N - 1)
        new_x_min_idx = tf.clip_by_value(new_x_min_idx, 0, N - 1)
        new_y_max_idx = tf.clip_by_value(new_y_max_idx, 0, N - 1)
        new_x_max_idx = tf.clip_by_value(new_x_max_idx, 0, N - 1)
        
        # Reformat as loc tokens
        new_y_min_token = tf.strings.join(["<loc", tf.strings.as_string(new_y_min_idx, width=4, fill="0"), ">"])
        new_x_min_token = tf.strings.join(["<loc", tf.strings.as_string(new_x_min_idx, width=4, fill="0"), ">"])
        new_y_max_token = tf.strings.join(["<loc", tf.strings.as_string(new_y_max_idx, width=4, fill="0"), ">"])
        new_x_max_token = tf.strings.join(["<loc", tf.strings.as_string(new_x_max_idx, width=4, fill="0"), ">"])
        
        return tf.strings.join([new_y_min_token, new_x_min_token, new_y_max_token, new_x_max_token])
    
    # Check if loc_tokens is empty
    is_empty = tf.equal(tf.strings.length(loc_tokens), 0)
    return tf.cond(is_empty, lambda: loc_tokens, extract_numbers)


def transform_bbox_for_letterbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float, float]:
    """Transform bbox coordinates for letterbox (resize with padding) transformation.

    When an image is resized with letterbox (maintaining aspect ratio with padding),
    the bbox coordinates need to be transformed to account for the scaling and padding.

    Args:
        x_min, y_min, x_max, y_max: Original normalized bbox coordinates (0-1).
        orig_w, orig_h: Original image dimensions.
        target_w, target_h: Target image dimensions after letterbox.

    Returns:
        Tuple of transformed (x_min, y_min, x_max, y_max) coordinates (0-1).
    """
    # Compute letterbox transformation parameters
    ratio = max(orig_w / target_w, orig_h / target_h)
    resized_w = int(orig_w / ratio)
    resized_h = int(orig_h / ratio)
    pad_w = (target_w - resized_w) / 2.0
    pad_h = (target_h - resized_h) / 2.0

    # Transform bbox coordinates
    new_x_min = x_min * (resized_w / target_w) + (pad_w / target_w)
    new_y_min = y_min * (resized_h / target_h) + (pad_h / target_h)
    new_x_max = x_max * (resized_w / target_w) + (pad_w / target_w)
    new_y_max = y_max * (resized_h / target_h) + (pad_h / target_h)

    # Clamp to valid range
    new_x_min = max(0.0, min(1.0, new_x_min))
    new_y_min = max(0.0, min(1.0, new_y_min))
    new_x_max = max(0.0, min(1.0, new_x_max))
    new_y_max = max(0.0, min(1.0, new_y_max))

    return new_x_min, new_y_min, new_x_max, new_y_max


def format_bbox_caption(
    objects: list[dict],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    apply_letterbox: bool = True,
) -> tuple[str, str]:
    """Format a list of objects with bboxes into prompt labels and caption.

    Args:
        objects: List of dicts with 'label' and 'bbox' ([x_min, y_min, x_max, y_max] normalized).
        orig_w, orig_h: Original image dimensions.
        target_w, target_h: Target image dimensions.
        apply_letterbox: Whether to apply letterbox transformation to bbox coordinates.

    Returns:
        Tuple of (prompt_labels, caption) strings.
    """
    if not objects:
        return "", ""

    # Build prompt with all object labels
    labels = [obj["label"] for obj in objects]
    unique_labels = list(dict.fromkeys(labels))  # Preserve order, remove duplicates
    prompt_labels = ", ".join(unique_labels)

    # Build caption with all bboxes
    caption_parts = []
    for obj in objects:
        label = obj["label"]
        bbox = obj["bbox"]  # [x_min, y_min, x_max, y_max] normalized 0-1

        if apply_letterbox:
            x_min, y_min, x_max, y_max = transform_bbox_for_letterbox(
                bbox[0], bbox[1], bbox[2], bbox[3],
                orig_w, orig_h, target_w, target_h
            )
        else:
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

        loc_str = bbox_to_loc_tokens(x_min, y_min, x_max, y_max)
        caption_parts.append(f"{loc_str} {label}")

    caption = " ; ".join(caption_parts)

    return prompt_labels, caption
