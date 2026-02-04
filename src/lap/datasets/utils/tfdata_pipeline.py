"""Generic dataset utilities for RLDS datasets."""

import logging
import os

import psutil
import tensorflow as tf

from lap.datasets.utils.image_utils import make_decode_images_fn


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


def gather_with_padding(
    data: tf.Tensor,
    sequence_length: tf.Tensor,
    window_size: int | tf.Tensor,
    per_timestep_windows: tf.Tensor | None = None,
) -> tf.Tensor:
    """Gather sliding windows with proper zero-padding (not repetition).

    This function replaces the buggy compute_window_indices approach that would
    repeat the last element instead of zero-padding.

    Args:
        data: Source tensor to gather from, shape [T, ...] where T is sequence length
        sequence_length: Scalar tensor, length of the sequence
        window_size: Scalar or tensor, size of the window to gather. If per_timestep_windows
                    is provided, this should be the maximum window size.
        per_timestep_windows: Optional [T] tensor specifying variable window size per timestep.
                             If None, uses fixed window_size for all timesteps.

    Returns:
        Gathered windows with shape [T, window_size, ...], properly zero-padded.
    """
    # Create base indices [T, window_size]
    if isinstance(window_size, int):
        window_size_tensor = tf.constant(window_size, dtype=tf.int32)
    else:
        window_size_tensor = tf.cast(window_size, tf.int32)

    base = tf.broadcast_to(tf.range(window_size_tensor)[None], [sequence_length, window_size_tensor])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size_tensor])
    indices = base + offsets  # [T, window_size], can exceed sequence_length - 1

    # Create validity mask
    if per_timestep_windows is not None:
        # Variable window sizes: check both sequence bounds and per-timestep window size
        sequence_valid = indices < sequence_length  # [T, window_size]
        window_valid = base < tf.expand_dims(per_timestep_windows, -1)  # [T, window_size]
        valid_mask = tf.logical_and(sequence_valid, window_valid)
    else:
        # Fixed window size: just check sequence bounds
        valid_mask = indices < sequence_length  # [T, window_size]

    # Clamp indices for gathering (to avoid TF errors)
    clamped_indices = tf.minimum(indices, sequence_length - 1)

    # Gather data
    gathered = tf.gather(data, clamped_indices)  # [T, window_size, ...]

    # Zero out invalid positions
    # Expand mask to match gathered shape
    mask_expanded = tf.cast(valid_mask, gathered.dtype)
    if len(gathered.shape) > 2:  # Has additional dimensions beyond [T, window]
        for _ in range(len(gathered.shape) - 2):
            mask_expanded = tf.expand_dims(mask_expanded, -1)

    return gathered * mask_expanded


def gather_with_last_value_padding(
    data: tf.Tensor,
    sequence_length: tf.Tensor,
    window_size: int | tf.Tensor,
    per_timestep_windows: tf.Tensor | None = None,
) -> tf.Tensor:
    """Gather sliding windows repeating the last valid value (no zero-padding).

    Args:
        data: Source tensor to gather from, shape [T, ...] where T is sequence length
        sequence_length: Scalar tensor, length of the sequence
        window_size: Scalar or tensor, size of the window to gather. If per_timestep_windows
                    is provided, this should be the maximum window size.
        per_timestep_windows: Optional [T] tensor specifying variable window size per timestep.
                             If None, uses fixed window_size for all timesteps.

    Returns:
        Gathered windows with shape [T, window_size, ...], padded by repeating
        the last valid value.
    """
    if isinstance(window_size, int):
        window_size_tensor = tf.constant(window_size, dtype=tf.int32)
    else:
        window_size_tensor = tf.cast(window_size, tf.int32)

    base = tf.broadcast_to(tf.range(window_size_tensor)[None], [sequence_length, window_size_tensor])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size_tensor])

    if per_timestep_windows is not None:
        max_base = tf.expand_dims(per_timestep_windows - 1, -1)
        base = tf.minimum(base, max_base)

    indices = base + offsets  # [T, window_size]
    clamped_indices = tf.minimum(indices, sequence_length - 1)

    return tf.gather(data, clamped_indices)


def dataset_size(ds: tf.data.Dataset) -> int:
    """Helper: try cardinality; fall back to counting if UNKNOWN/INFINITE."""
    c = ds.cardinality().numpy()  # returns int64 or negative sentinel
    if c >= 0:
        return int(c)
    # Count explicitly (works after .filter/.flat_map, etc.)
    return int(ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())


def prepare_batched_dataset(
    dataset,
    want_val,
    shuffle,
    shuffle_buffer_size,
    seed,
    max_samples,
    batch_size,
    resize_resolution,
    primary_image_key,
    wrist_image_key,
    wrist_image_right_key=None,
    *,
    aggressive_aug: bool = False,
    aug_wrist_image: bool = True,
    not_rotate_wrist_prob: float = 0.0,
):
    """Prepare a batched dataset with optional aggressive augmentation.

    Args:
        dataset: The input TensorFlow dataset.
        want_val: Whether this is for validation.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Size of the shuffle buffer.
        seed: Random seed for shuffling.
        max_samples: Maximum number of samples to use.
        batch_size: Batch size.
        resize_resolution: Target resolution for resizing images.
        primary_image_key: Key for the primary (base) image.
        wrist_image_key: Key for the wrist image.
        wrist_image_right_key: Optional key for right wrist image.
        aggressive_aug: If True, apply aggressive augmentation BEFORE padding.
            This mirrors the logic from preprocess_observation_aggressive and
            makes cropping more effective since it operates on original images.
            Only applied during training (when want_val=False) and only for samples
            where dataset_name contains "droid".
        aug_wrist_image: If True and aggressive_aug is True, augment wrist images.
        not_rotate_wrist_prob: Probability of NOT rotating wrist images for samples
            that have needs_wrist_rotation=True. Set to 0.0 for validation.
    """
    # Apply standard pipeline operations
    if want_val:
        # Validation: iterate once through the data
        if max_samples is not None:
            dataset = dataset.take(int(max_samples))
        # NOTE: For validation, we don't cache since we reconstruct iterator each time
        # and max_samples is typically not used, so we iterate until StopIteration
    elif (not want_val) and shuffle and max_samples is None:
        # Training with shuffling: repeat and shuffle
        dataset = dataset.repeat().shuffle(shuffle_buffer_size, seed=seed)
    elif (not want_val) and not shuffle and max_samples is None:
        # Training without shuffling: just iterate once (no repeat, no shuffle)
        # This is useful for evaluation on training split
        pass  # Don't modify dataset, just iterate once
    elif (not want_val) and max_samples is not None:
        # Training with max_samples limit
        dataset = dataset.take(int(max_samples))
    else:
        raise NotImplementedError(
            f"Unsupported mode: want_val={want_val}, shuffle={shuffle}, max_samples={max_samples}"
        )

    # Only apply aggressive augmentation during training (not validation)
    # Per-sample check for "droid" in dataset_name is done inside make_decode_images_fn
    apply_aggressive_aug = aggressive_aug and (not want_val)
    # Only apply random rotation skip during training (not validation)
    effective_not_rotate_prob = not_rotate_wrist_prob if not want_val else 0.0

    decode_fn = make_decode_images_fn(
        primary_key=primary_image_key,
        wrist_key=wrist_image_key,
        wrist_right_key=wrist_image_right_key,
        resize_to=resize_resolution,
        aggressive_aug=apply_aggressive_aug,
        aug_wrist_image=aug_wrist_image,
        not_rotate_wrist_prob=effective_not_rotate_prob,
        seed=seed,
    )
    num_parallel_calls = tf.data.AUTOTUNE
    dataset = dataset.frame_map(decode_fn, num_parallel_calls)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Apply device-specific and buffering operations
    # For validation, use smaller prefetch buffer since we only iterate once
    # and don't need aggressive buffering
    if want_val:
        # Simplified buffering for validation - just basic prefetch
        try:
            dataset = dataset.prefetch_to_device(1)  # Smaller buffer for validation
        except Exception:
            dataset = dataset.prefetch(1)
        # Skip with_ram_budget for validation to avoid expensive memory management
        # since we're only iterating once
    else:
        # Full buffering for training
        try:
            dataset = dataset.prefetch_to_device(2)
        except Exception:
            dataset = dataset.prefetch(2)
        dataset = dataset.with_ram_budget(1)

    return dataset
