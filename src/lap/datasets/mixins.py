"""Reusable mixins for dataset classes.

This module provides mixins that encapsulate common functionality shared
across robot and VQA datasets, reducing code duplication and ensuring
consistent behavior.

Usage:
    class MyDataset(TFConfigMixin, DatasetOptionsMixin, ValidationSplitMixin):
        def __init__(self, ...):
            self.configure_tf_devices()
            ...
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import psutil
import tensorflow as tf

if TYPE_CHECKING:
    pass


class TFConfigMixin:
    """Mixin to configure TensorFlow to avoid GPU/TPU clobbering with JAX.

    Call `configure_tf_devices()` early in __init__ to ensure TensorFlow
    doesn't allocate GPU/TPU memory that should be reserved for JAX.
    """

    def configure_tf_devices(self) -> None:
        """Configure TensorFlow with no GPU/TPU devices."""
        tf.config.set_visible_devices([], "GPU")
        with contextlib.suppress(Exception):
            tf.config.set_visible_devices([], "TPU")


class DatasetOptionsMixin:
    """Mixin providing optimized tf.data.Options for dataset pipelines.

    Provides consistent performance tuning across all dataset types.
    """

    def get_dataset_ops(self, want_full_determinism: bool = False) -> tf.data.Options:
        """Get optimized tf.data.Options for the dataset pipeline.

        Args:
            want_full_determinism: If True, enable deterministic operations
                for reproducibility. Usually True for validation.

        Returns:
            Configured tf.data.Options object.
        """
        opts = tf.data.Options()

        # Determinism settings
        opts.experimental_deterministic = want_full_determinism

        # Autotune settings
        opts.autotune.enabled = True

        # Optimization settings
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_fusion = True
        opts.experimental_optimization.map_and_filter_fusion = True
        opts.experimental_optimization.inject_prefetch = False
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True

        # Threading settings
        opts.experimental_warm_start = True
        opts.experimental_threading.private_threadpool_size = int(max(16, psutil.cpu_count(logical=True)))

        return opts


class ValidationSplitMixin:
    """Mixin for consistent train/val splitting using hash-based bucketing.

    Uses deterministic hashing to assign trajectories/samples to train or
    validation splits, ensuring consistency across runs and processes.
    """

    def make_split_filter(
        self,
        id_key: str,
        split_seed: int,
        val_fraction: float,
        want_val: bool,
    ):
        """Create a filter function for train/val splitting.

        Args:
            id_key: Key to access the trajectory/sample ID for hashing.
            split_seed: Seed for consistent splitting across runs.
            val_fraction: Fraction of data to use for validation (0.0-1.0).
            want_val: If True, keep validation samples; else keep training.

        Returns:
            Filter function suitable for dataset.filter().
        """

        def _split_filter(data):
            salt = tf.strings.as_string(split_seed)
            # Handle both trajectory (repeated ID) and frame (scalar ID) formats
            anchor = data[id_key]
            if anchor.shape.ndims > 0:
                anchor = anchor[0]
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if want_val else tf.logical_not(is_val)

        return _split_filter

    def apply_split_filter(
        self,
        dataset,
        id_key: str,
        split_seed: int,
        val_fraction: float,
        want_val: bool,
    ):
        """Apply train/val split filter to a dataset.

        Args:
            dataset: The dataset to filter.
            id_key: Key to access the trajectory/sample ID for hashing.
            split_seed: Seed for consistent splitting across runs.
            val_fraction: Fraction of data to use for validation (0.0-1.0).
            want_val: If True, keep validation samples; else keep training.

        Returns:
            Filtered dataset.
        """
        filter_fn = self.make_split_filter(id_key, split_seed, val_fraction, want_val)
        return dataset.filter(filter_fn)


class DummyStatisticsMixin:
    """Mixin for creating dummy normalization statistics.

    Used by VQA and other datasets that don't have meaningful action/state
    statistics but need to provide them for compatibility with the training
    pipeline.
    """

    def build_dummy_statistics(
        self,
        action_dim: int,
        state_dim: int,
        num_transitions: int,
        num_trajectories: int = 0,
    ) -> dict:
        """Build dummy normalization statistics.

        Creates statistics with zero mean, unit std, and appropriate shapes
        for compatibility with normalization transforms.

        Args:
            action_dim: Dimension of action space.
            state_dim: Dimension of state space.
            num_transitions: Number of transitions (frames) in the dataset.
            num_trajectories: Number of trajectories (optional).

        Returns:
            Dictionary with 'actions' and 'state' ExtendedNormStats.
        """
        from lap.shared.normalize_adapter import ExtendedNormStats

        return {
            "actions": ExtendedNormStats(
                mean=np.zeros(action_dim, dtype=np.float32),
                std=np.ones(action_dim, dtype=np.float32),
                q01=np.zeros(action_dim, dtype=np.float32),
                q99=np.zeros(action_dim, dtype=np.float32),
                num_transitions=num_transitions,
                num_trajectories=num_trajectories,
            ),
            "state": ExtendedNormStats(
                mean=np.zeros(state_dim, dtype=np.float32),
                std=np.ones(state_dim, dtype=np.float32),
                q01=np.zeros(state_dim, dtype=np.float32),
                q99=np.zeros(state_dim, dtype=np.float32),
                num_transitions=num_transitions,
                num_trajectories=num_trajectories,
            ),
        }


class RLDSDatasetMixin(TFConfigMixin, DatasetOptionsMixin):
    """Mixin for RLDS-based dataset loading.

    Provides common functionality for loading datasets from RLDS format
    with proper sharding for multi-process training.
    """

    def build_rlds_dataset(
        self,
        builder,
        shuffle: bool,
        num_parallel_reads: int,
        shard_for_training: bool = True,
    ):
        """Build a DLataset from an RLDS builder.

        Args:
            builder: TFDS builder for the RLDS dataset.
            shuffle: Whether to shuffle at file/shard level.
            num_parallel_reads: Number of parallel file reads.
            shard_for_training: If True, shard across JAX processes.
                Usually True for training, False for validation.

        Returns:
            Configured DLataset.
        """
        import dlimp as dl
        import jax

        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=shuffle,
            num_parallel_reads=num_parallel_reads,
        )

        # Shard across processes for distributed training
        if shard_for_training:
            dataset = dataset.shard(jax.process_count(), jax.process_index())

        return dataset


class TrajectoryIdentifierMixin:
    """Mixin for generating trajectory identifiers.

    Provides hash-based trajectory ID generation that is consistent
    across runs while being unique per trajectory.
    """

    def compute_trajectory_hash(
        self,
        action_tensor: tf.Tensor,
        dataset_name: str,
        seed: int,
        max_steps: int = 128,
    ) -> tf.Tensor:
        """Compute a hash-based trajectory identifier.

        Uses the action sequence to generate a unique, deterministic ID.
        For long trajectories, uses first/last 64 steps for efficiency.

        Args:
            action_tensor: Action tensor of shape [T, action_dim].
            dataset_name: Name of the dataset (included in hash).
            seed: Random seed for hash function.
            max_steps: Maximum steps to use for hashing.

        Returns:
            String tensor with trajectory ID like "dataset_name-123456789".
        """
        traj_len = tf.shape(action_tensor)[0]

        # Use first/last 64 steps for long trajectories
        action_for_hash = tf.cond(
            max_steps >= traj_len,
            lambda: action_tensor,
            lambda: tf.concat([action_tensor[:64], action_tensor[-64:]], axis=0),
        )

        serialized_action = tf.io.serialize_tensor(action_for_hash)
        name_tensor = tf.constant(dataset_name, dtype=tf.string)
        to_hash = tf.strings.join([name_tensor, tf.constant("::", dtype=tf.string), serialized_action])
        hashed = tf.strings.to_hash_bucket_strong(to_hash, 2147483647, key=[seed, 1337])

        return tf.strings.join(
            [
                name_tensor,
                tf.constant("-", dtype=tf.string),
                tf.strings.as_string(hashed),
            ]
        )


class StandaloneBatchingMixin:
    """Mixin for standalone dataset batching and augmentation.

    Provides the final batching, shuffling, and augmentation steps
    for datasets that will be used standalone (not mixed).
    """

    def apply_standalone_batching(
        self,
        dataset,
        config,
        want_val: bool,
        shuffle: bool,
        seed: int,
        max_samples: int | None,
        batch_size: int,
        primary_image_key: str,
        wrist_image_key: str,
        wrist_image_right_key: str | None = None,
    ):
        """Apply final batching and augmentation for standalone use.

        Args:
            dataset: The dataset to batch.
            config: Data configuration with augmentation settings.
            want_val: If True, disable training augmentations.
            shuffle: Whether to shuffle before batching.
            seed: Random seed for shuffling.
            max_samples: Optional limit on number of samples.
            batch_size: Batch size for output.
            primary_image_key: Key for primary image in observations.
            wrist_image_key: Key for wrist image in observations.
            wrist_image_right_key: Optional key for right wrist image.

        Returns:
            Batched and optionally augmented dataset.
        """
        from lap.datasets.utils.tfdata_pipeline import prepare_batched_dataset

        return prepare_batched_dataset(
            dataset=dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
            batch_size=batch_size,
            resize_resolution=config.resize_resolution,
            primary_image_key=primary_image_key,
            wrist_image_key=wrist_image_key,
            wrist_image_right_key=wrist_image_right_key,
            aggressive_aug=getattr(config, "aggressive_aug", False),
            aug_wrist_image=getattr(config, "aug_wrist_image", True),
            not_rotate_wrist_prob=getattr(config, "not_rotate_wrist_prob", 0.0),
        )
