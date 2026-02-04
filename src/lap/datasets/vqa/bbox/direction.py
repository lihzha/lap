"""Direction classification utilities for VQA datasets.

This module provides functions for computing and manipulating direction
labels based on bounding box positions relative to image center.
"""

import tensorflow as tf


def direction_from_bbox_tf(
    bbox: tf.Tensor,
    slope: float = 2.0,
    add_move_prefix: bool = False,
) -> tf.Tensor:
    """Map bbox center to direction string relative to image center.

    Args:
        bbox: Tensor of shape [2, 2] with [[x_min, y_min], [x_max, y_max]].
        slope: Slope parameter for direction boundaries (default 2.0).
        add_move_prefix: If True, prefix direction with "move " (e.g., "move left").

    Returns:
        tf.string tensor with direction ("forward", "back", "left", "right",
        or compound like "left and forward"). If add_move_prefix is True,
        returns "move forward", "move left and forward", etc.
    """
    top_left = bbox[0]
    bottom_right = bbox[1]
    center = (top_left + bottom_right) / 2.0

    x_rel = center[0] - 0.5  # +x is right
    y_rel = 0.5 - center[1]  # invert so +y is up

    k = tf.constant(slope, dtype=tf.float32)
    inv_k = 1.0 / k

    abs_x = tf.abs(x_rel)
    abs_y = tf.abs(y_rel)

    # Primary axis regions using slopes k and 1/k
    is_forward = y_rel > k * abs_x
    is_back = y_rel < -k * abs_x
    is_right = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
    is_right = tf.logical_and(is_right, x_rel > inv_k * abs_y)
    is_left = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
    is_left = tf.logical_and(is_left, x_rel < -inv_k * abs_y)

    def forward():
        return tf.constant("forward")

    def back():
        return tf.constant("back")

    def right():
        return tf.constant("right")

    def left():
        return tf.constant("left")

    def diagonal():
        base_dir = tf.where(x_rel < 0.0, tf.constant("left"), tf.constant("right"))
        vert_dir = tf.where(y_rel >= 0.0, tf.constant("forward"), tf.constant("back"))
        return tf.strings.join([base_dir, " and ", vert_dir])

    direction = tf.case(
        [
            (is_forward, forward),
            (is_back, back),
            (is_right, right),
            (is_left, left),
        ],
        default=diagonal,
        exclusive=True,
    )

    if add_move_prefix:
        return tf.strings.join(["move ", direction])
    return direction


def rotate_direction_180_tf(direction: tf.Tensor) -> tf.Tensor:
    """Rotate direction string by 180 degrees.
    
    For a 180-degree rotation:
    - "forward" -> "back"
    - "back" -> "forward"
    - "left" -> "right"
    - "right" -> "left"
    - "left and forward" -> "right and back"
    - "right and forward" -> "left and back"
    - "left and back" -> "right and forward"
    - "right and back" -> "left and forward"
    
    Also handles "move " prefix: "move forward" -> "move back"
    
    Args:
        direction: tf.string tensor with direction (e.g., "move forward", "left and back").
        
    Returns:
        tf.string tensor with rotated direction.
    """
    # Check if direction has "move " prefix
    has_move_prefix = tf.strings.regex_full_match(direction, r"move .*")
    
    def remove_prefix():
        return tf.strings.regex_replace(direction, r"^move ", "")
    
    def add_prefix(d):
        return tf.strings.join(["move ", d])
    
    direction_no_prefix = tf.cond(has_move_prefix, remove_prefix, lambda: direction)
    
    # Rotate using string replacements (works for both simple and compound)
    # Replace left/right with placeholders to avoid double replacement
    rotated = tf.strings.regex_replace(direction_no_prefix, r"\bleft\b", "__LEFT_PLACEHOLDER__")
    rotated = tf.strings.regex_replace(rotated, r"\bright\b", "left")
    rotated = tf.strings.regex_replace(rotated, r"__LEFT_PLACEHOLDER__", "right")
    
    # Replace forward/back
    rotated = tf.strings.regex_replace(rotated, r"\bforward\b", "__FORWARD_PLACEHOLDER__")
    rotated = tf.strings.regex_replace(rotated, r"\bback\b", "forward")
    rotated = tf.strings.regex_replace(rotated, r"__FORWARD_PLACEHOLDER__", "back")
    
    # Add back "move " prefix if it was there
    final_direction = tf.cond(
        has_move_prefix,
        lambda: add_prefix(rotated),
        lambda: rotated,
    )
    
    return final_direction


def compute_direction_from_bbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    slope: float = 2.0,
    add_move_prefix: bool = True,
) -> str:
    """Compute direction string from bbox coordinates (Python version for table building).

    Args:
        x_min, y_min, x_max, y_max: Normalized bbox coordinates (0-1).
        slope: Slope parameter for direction boundaries (default 2.0).
        add_move_prefix: If True, prefix direction with "move " (e.g., "move left").

    Returns:
        Direction string. If add_move_prefix is True (default), returns "move forward",
        "move left and forward", etc. Otherwise returns just "forward", "left and forward", etc.
    """
    # Compute center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    x_rel = center_x - 0.5  # +x is right
    y_rel = 0.5 - center_y  # invert so +y is up/forward

    k = slope
    inv_k = 1.0 / k

    abs_x = abs(x_rel)
    abs_y = abs(y_rel)

    # Primary axis regions using slopes k and 1/k
    is_forward = y_rel > k * abs_x
    is_back = y_rel < -k * abs_x
    is_right = (not is_forward) and (not is_back) and (x_rel > inv_k * abs_y)
    is_left = (not is_forward) and (not is_back) and (x_rel < -inv_k * abs_y)

    if is_forward:
        direction = "forward"
    elif is_back:
        direction = "back"
    elif is_right:
        direction = "right"
    elif is_left:
        direction = "left"
    else:
        # Diagonal
        base_dir = "left" if x_rel < 0.0 else "right"
        vert_dir = "forward" if y_rel >= 0.0 else "back"
        direction = f"{base_dir} and {vert_dir}"

    if add_move_prefix:
        # With probability 0.5, add or not add "move " prefix
        import tensorflow as tf
        prob = tf.random.uniform([], dtype=tf.float32)
        if prob < 0.5:
            return f"move {direction}"
    return direction
