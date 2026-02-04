"""Image processing utilities for RLDS datasets."""

from __future__ import annotations

import tensorflow as tf


def tf_rotate_180(image: tf.Tensor) -> tf.Tensor:
    """Rotate image by 180 degrees (equivalent to np.rot90(image, k=2)).

    Args:
        image: Image tensor of shape [H, W, C] or [T, H, W, C]

    Returns:
        Rotated image tensor with same shape
    """
    # tf.image.rot90 with k=2 rotates 180 degrees
    # Handle both single image [H, W, C] and batched [T, H, W, C] cases
    if len(image.shape) == 4:
        # Batched case [T, H, W, C]
        return tf.map_fn(lambda img: tf.image.rot90(img, k=2), image)
    # Single image [H, W, C]
    return tf.image.rot90(image, k=2)


def tf_maybe_rotate_180(
    image: tf.Tensor,
    should_rotate: tf.Tensor,
    seed_pair: tuple[int, int] | None = None,
    not_rotate_prob: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Conditionally rotate image by 180 degrees with optional randomization.

    Args:
        image: Image tensor of shape [H, W, C] or [T, H, W, C]
        should_rotate: Boolean tensor indicating if dataset requires rotation
        seed_pair: Optional (seed1, seed2) for reproducible random decision
        not_rotate_prob: Probability of NOT rotating even when should_rotate=True

    Returns:
        Tuple of (rotated_image, did_rotate_bool) where did_rotate_bool indicates
        if rotation was actually applied (for downstream EEF frame adjustments)
    """

    def do_rotate():
        if not_rotate_prob > 0.0 and seed_pair is not None:
            # Use stateless random for reproducibility
            rand_val = tf.random.stateless_uniform([], seed=seed_pair, dtype=tf.float32)
            skip_rotation = rand_val < not_rotate_prob
            return tf.cond(
                skip_rotation,
                lambda: (image, tf.constant(False)),
                lambda: (tf_rotate_180(image), tf.constant(True)),
            )
        if not_rotate_prob > 0.0:
            # Use regular random
            rand_val = tf.random.uniform([], dtype=tf.float32)
            skip_rotation = rand_val < not_rotate_prob
            return tf.cond(
                skip_rotation,
                lambda: (image, tf.constant(False)),
                lambda: (tf_rotate_180(image), tf.constant(True)),
            )
        # Always rotate
        return (tf_rotate_180(image), tf.constant(True))

    def no_rotate():
        return (image, tf.constant(False))

    return tf.cond(should_rotate, do_rotate, no_rotate)


def _tf_aggressive_augment(
    image: tf.Tensor,
    height_crop_frac: float | tf.Tensor = 0.99,
    width_crop_frac: float = 0.9,
    use_random_height_frac: bool = False,
    seed: tf.Tensor | None = None,
) -> tf.Tensor:
    """Apply aggressive augmentation to images BEFORE padding.

    This is a unified augmentation function that handles both wrist and base images
    with different crop parameters.

    Args:
        image: Input image tensor [H, W, C]
        height_crop_frac: Fixed height crop fraction, or ignored if use_random_height_frac
        width_crop_frac: Width crop fraction (default 0.9)
        use_random_height_frac: If True, randomly sample height crop from [0.65] * 8
        seed: Random seed for reproducibility

    Returns:
        Augmented image tensor with same shape
    """
    orig_h = tf.shape(image)[0]
    orig_w = tf.shape(image)[1]
    orig_dtype = image.dtype

    # Work in float32 for augmentation
    if orig_dtype == tf.uint8:
        image = tf.cast(image, tf.float32) / 255.0
    else:
        # Assume [-1, 1] range, convert to [0, 1]
        image = image / 2.0 + 0.5

    # Determine crop height fraction
    if use_random_height_frac:
        crop_fracs = tf.constant([0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65], dtype=tf.float32)
        crop_idx = tf.random.uniform([], 0, 8, dtype=tf.int32, seed=seed)
        height_frac = tf.gather(crop_fracs, crop_idx)
    else:
        height_frac = tf.cast(height_crop_frac, tf.float32)

    crop_h = tf.cast(tf.cast(orig_h, tf.float32) * height_frac, tf.int32)
    crop_w = tf.cast(tf.cast(orig_w, tf.float32) * width_crop_frac, tf.int32)

    # Random crop
    image = tf.image.random_crop(image, [crop_h, crop_w, 3], seed=seed)

    # Resize back to original dimensions
    image = tf.image.resize(image, [orig_h, orig_w], method=tf.image.ResizeMethod.BILINEAR)

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Convert back to original dtype
    if orig_dtype == tf.uint8:
        image = tf.cast(image * 255.0, tf.uint8)
    else:
        # Convert back to [-1, 1]
        image = image * 2.0 - 1.0

    return image


def _tf_aggressive_augment_wrist(image: tf.Tensor, seed: tf.Tensor | None = None) -> tf.Tensor:
    """Apply aggressive augmentation to wrist images BEFORE padding.

    Uses random height crop fraction and 0.9 width crop.
    """
    return _tf_aggressive_augment(
        image,
        use_random_height_frac=True,
        width_crop_frac=0.9,
        seed=seed,
    )


def _tf_aggressive_augment_base(image: tf.Tensor, seed: tf.Tensor | None = None) -> tf.Tensor:
    """Apply aggressive augmentation to base (non-wrist) images BEFORE padding.

    Uses 0.99 height crop and 0.9 width crop.
    """
    return _tf_aggressive_augment(
        image,
        height_crop_frac=0.99,
        width_crop_frac=0.9,
        use_random_height_frac=False,
        seed=seed,
    )


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    wrist_right_key: str | None = None,
    resize_to: tuple[int, int] | None = (224, 224),
    aggressive_aug: bool = False,
    aug_wrist_image: bool = True,
    not_rotate_wrist_prob: float = 0.0,
    seed: int = 0,
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors.
    Preserves aspect ratio, pads symmetrically, and returns the original dtype semantics
    (uint8 clamped 0-255, float32 clamped to [-1, 1]).

    Args:
        primary_key: Key for the primary (base) image in the observation dict.
        wrist_key: Key for the wrist image in the observation dict.
        wrist_right_key: Optional key for right wrist image.
        resize_to: Target resolution (height, width) for resizing with padding.
        aggressive_aug: If True, apply aggressive augmentation BEFORE padding.
            This mirrors the logic from preprocess_observation_aggressive and
            makes cropping more effective since it operates on original images.
        aug_wrist_image: If True and aggressive_aug is True, augment wrist images.
        not_rotate_wrist_prob: Probability of NOT rotating wrist images even when
            rotation is required (samples with needs_wrist_rotation=True).
        seed: Random seed for reproducible rotation decisions.
    """

    def _tf_resize_with_pad(image: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        # Compute resized dimensions preserving aspect ratio
        in_h = tf.shape(image)[0]
        in_w = tf.shape(image)[1]
        orig_dtype = image.dtype

        h_f = tf.cast(in_h, tf.float32)
        w_f = tf.cast(in_w, tf.float32)
        th_f = tf.cast(target_h, tf.float32)
        tw_f = tf.cast(target_w, tf.float32)

        ratio = tf.maximum(w_f / tw_f, h_f / th_f)
        resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
        resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)

        # Resize in float32
        img_f32 = tf.cast(image, tf.float32)
        resized_f32 = tf.image.resize(img_f32, [resized_h, resized_w], method=tf.image.ResizeMethod.BILINEAR)

        # Dtype-specific postprocess (python conditional on static dtype)
        if orig_dtype == tf.uint8:
            resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
            const_val = tf.constant(0, dtype=resized.dtype)
        else:
            resized = tf.clip_by_value(resized_f32, -1.0, 1.0)
            const_val = tf.constant(-1.0, dtype=resized.dtype)

        # Compute symmetric padding
        pad_h_total = target_h - resized_h
        pad_w_total = target_w - resized_w
        pad_h0 = pad_h_total // 2
        pad_h1 = pad_h_total - pad_h0
        pad_w0 = pad_w_total // 2
        pad_w1 = pad_w_total - pad_w0

        padded = tf.pad(resized, [[pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], constant_values=const_val)
        return padded

    def _decode_single(img_bytes, is_wrist: bool = False, apply_aug: bool = False):
        """Decode image bytes and optionally apply augmentation before padding.

        Args:
            img_bytes: Encoded image bytes or numeric tensor.
            is_wrist: Whether this is a wrist image (affects augmentation).
            apply_aug: Whether to apply augmentation to this specific image.
        """
        # If already numeric, cast to uint8 and return
        if img_bytes.dtype != tf.string:
            img = tf.cast(img_bytes, tf.uint8)
        else:
            # Guard against empty placeholders (e.g., padding "")
            has_data = tf.greater(tf.strings.length(img_bytes), 0)
            img = tf.cond(
                has_data,
                lambda: tf.io.decode_image(
                    img_bytes,
                    channels=3,
                    expand_animations=False,
                    dtype=tf.uint8,
                ),
                lambda: tf.zeros([1, 1, 3], dtype=tf.uint8),
            )

        # Apply aggressive augmentation BEFORE padding (if enabled and this sample should be augmented)
        # This makes cropping more effective since it operates on original images
        if aggressive_aug and apply_aug:
            if is_wrist and aug_wrist_image:
                img = _tf_aggressive_augment_wrist(img)
            elif not is_wrist:
                img = _tf_aggressive_augment_base(img)

        # Optional resize-with-pad to ensure batching shape compatibility
        if resize_to is not None:
            h, w = resize_to
            img = _tf_resize_with_pad(img, h, w)
        return img

    def _decode_frame(traj: dict) -> dict:
        # Check if this sample is from DROID dataset (only augment DROID samples)
        # dataset_name is stored in the trajectory dict
        dataset_name = traj.get("dataset_name", tf.constant("", dtype=tf.string))
        # Check if "droid" is in the dataset name (case-insensitive)
        is_droid = tf.strings.regex_full_match(tf.strings.lower(dataset_name), ".*droid.*")

        # Use tf.cond to conditionally apply augmentation
        def decode_with_aug():
            return (
                _decode_single(traj["observation"][primary_key], is_wrist=False, apply_aug=True),
                _decode_single(traj["observation"][wrist_key], is_wrist=True, apply_aug=True),
            )

        def decode_without_aug():
            return (
                _decode_single(traj["observation"][primary_key], is_wrist=False, apply_aug=False),
                _decode_single(traj["observation"][wrist_key], is_wrist=True, apply_aug=False),
            )

        primary_img, wrist_img = tf.cond(is_droid, decode_with_aug, decode_without_aug)

        # Apply wrist image rotation if sample requires it
        # This is used for DROID and other datasets with inverted wrist cameras
        needs_rotation = traj.get("needs_wrist_rotation", tf.constant(False, dtype=tf.bool))
        is_prediction_sample = traj.get("is_prediction_sample", tf.constant(False, dtype=tf.bool))
        pred_use_primary = traj.get("pred_use_primary", tf.constant(False, dtype=tf.bool))

        # Track whether rotation was actually applied (for EEF frame adjustment in lap_policy)
        rotation_applied = tf.constant(False, dtype=tf.bool)

        def maybe_rotate_wrist(img):
            """Apply 180-degree rotation with probability (1 - not_rotate_wrist_prob)."""
            if not_rotate_wrist_prob > 0.0:
                # Use random decision
                skip_rotation = tf.random.uniform([], dtype=tf.float32) < not_rotate_wrist_prob
                rotated_img = tf.cond(
                    skip_rotation,
                    lambda: img,
                    lambda: tf_rotate_180(img),
                )
                did_rotate = tf.logical_not(skip_rotation)
                return rotated_img, did_rotate
            # Always rotate
            return tf_rotate_180(img), tf.constant(True, dtype=tf.bool)

        # Rotation logic:
        # - For prediction samples using wrist camera: rotate both primary and wrist if needs_rotation
        # - For prediction samples using primary camera: don't rotate
        # - For non-prediction samples: rotate wrist if needs_rotation
        def handle_prediction_wrist():
            # Prediction sample using wrist camera: rotate both primary and wrist
            # Use the same rotation decision for both images
            if needs_rotation:
                if not_rotate_wrist_prob > 0.0:
                    # Use a single random decision for both images
                    skip_rotation = tf.random.uniform([], dtype=tf.float32) < not_rotate_wrist_prob

                    def rotate_both():
                        return tf_rotate_180(primary_img), tf_rotate_180(wrist_img), tf.constant(True, dtype=tf.bool)

                    def no_rotate_both():
                        return primary_img, wrist_img, tf.constant(False, dtype=tf.bool)

                    rotated_primary, rotated_wrist, did_rotate = tf.cond(
                        skip_rotation,
                        no_rotate_both,
                        rotate_both,
                    )
                    return rotated_primary, rotated_wrist, did_rotate
                # Always rotate both
                return tf_rotate_180(primary_img), tf_rotate_180(wrist_img), tf.constant(True, dtype=tf.bool)
            return primary_img, wrist_img, tf.constant(False, dtype=tf.bool)

        def handle_prediction_primary():
            # Prediction sample using primary camera: don't rotate
            return primary_img, wrist_img, tf.constant(False, dtype=tf.bool)

        def handle_regular():
            # Regular sample: rotate wrist if needs_rotation
            if needs_rotation:
                rotated_wrist, did_rotate = maybe_rotate_wrist(wrist_img)
                return primary_img, rotated_wrist, did_rotate
            return primary_img, wrist_img, tf.constant(False, dtype=tf.bool)

        # Determine which handler to use
        is_pred_wrist = tf.logical_and(is_prediction_sample, tf.logical_not(pred_use_primary))
        is_pred_primary = tf.logical_and(is_prediction_sample, pred_use_primary)

        primary_img, wrist_img, rotation_applied = tf.case(
            [
                (is_pred_wrist, handle_prediction_wrist),
                (is_pred_primary, handle_prediction_primary),
            ],
            default=handle_regular,
            exclusive=True,
        )

        traj["observation"][primary_key] = primary_img
        traj["observation"][wrist_key] = wrist_img
        # Track whether rotation was applied for EEF frame adjustment
        traj["rotation_applied"] = rotation_applied
        # traj["observation"][wrist_right_key] = _decode_single(traj["observation"][wrist_right_key])

        return traj

    return _decode_frame
