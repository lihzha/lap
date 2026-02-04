"""Base dataset classes for RLDS datasets."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import logging
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.dataset_configs import get_tfds_name_with_version
from lap.datasets.mixins import DatasetOptionsMixin
from lap.datasets.mixins import DummyStatisticsMixin
from lap.datasets.mixins import RLDSDatasetMixin
from lap.datasets.mixins import TFConfigMixin
from lap.datasets.mixins import TrajectoryIdentifierMixin
from lap.datasets.mixins import ValidationSplitMixin
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import StateEncoding
from lap.datasets.utils.normalization_and_config import load_dataset_kwargs
from lap.datasets.utils.rotation_utils import euler_diff
from lap.datasets.utils.rotation_utils import euler_to_r6 as euler_to_rot6d
from lap.datasets.utils.rotation_utils import euler_to_rotation_matrix as _R_from_euler_xyz
from lap.datasets.utils.specs import RldsDatasetSpec
from lap.datasets.utils.tfdata_pipeline import gather_with_last_value_padding
from lap.datasets.utils.tfdata_pipeline import gather_with_padding
from lap.datasets.utils.tfdata_pipeline import prepare_batched_dataset
from lap.shared.normalize_adapter import check_dataset_statistics
from lap.shared.normalize_adapter import get_dataset_statistics

if TYPE_CHECKING:
    from lap.training.config import DataConfig


class BaseDataset(ABC, TFConfigMixin, DatasetOptionsMixin, DummyStatisticsMixin):
    """Abstract base class for all datasets (robot and VQA).

    This class provides common functionality shared between robot datasets
    (BaseRobotDataset) and VQA datasets (BaseVQADataset):

    - TensorFlow device configuration (no GPU/TPU for JAX compatibility)
    - Parallel processing settings (num_parallel_reads, num_parallel_calls)
    - Train/validation splitting with consistent hashing
    - Statistics handling (both dummy and computed)
    - Standalone batching and augmentation
    - Iterator protocol implementation

    Subclasses must implement:
    - build_dataset_builder(): Create the TFDS builder
    - build_dataset(): Build the tf.data.Dataset
    - get_dataset_name(): Return the dataset name string
    """

    # Shared spec for image keys (can be overridden in subclasses)
    spec: ClassVar[RldsDatasetSpec] = RldsDatasetSpec()

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: DataConfig,
        action_dim: int = 32,
        state_dim: int = 10,
        action_horizon: int = 16,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE
        seed: int = 0,
        split: str = "train",
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    ):
        """Initialize common dataset attributes.

        Args:
            dataset_name: Name of the dataset (e.g., "droid", "coco_captions").
            data_dir: Path to the TFDS data directory.
            config: Data configuration object.
            action_dim: Dimension of action space (for padding).
            state_dim: Dimension of state space (for padding).
            action_horizon: Number of action steps to chunk.
            num_parallel_reads: Parallel file reads (-1 for AUTOTUNE).
            num_parallel_calls: Parallel map calls (-1 for AUTOTUNE).
            seed: Random seed for reproducibility.
            split: Dataset split ("train" or "val").
            standalone: If True, apply batching/augmentation for standalone use.
            shuffle: Whether to shuffle the dataset.
            batch_size: Batch size for standalone mode.
            max_samples: Maximum samples to use (None for all).
            action_proprio_normalization_type: Normalization scheme for actions/state.
        """
        # Store basic config
        self.config = config
        self.seed = seed
        self.want_val = split == "val"
        self.dataset_name = dataset_name
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.standalone = standalone
        self.want_full_determinism = bool(config.want_full_determinism)

        # Parallel processing settings
        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)

        # Store batching params for standalone mode
        self._standalone = standalone
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._max_samples = max_samples

        # Configure TensorFlow devices using mixin
        self.configure_tf_devices()
        tf.random.set_seed(seed)

        # Dataset and statistics will be set by subclasses
        self.dataset: dl.DLataset | tf.data.Dataset = None
        self.dataset_statistics: dict | None = None
        self.builder = None

    @abstractmethod
    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for this dataset.

        Args:
            ds_name: Dataset name.
            data_dir: Data directory path.

        Returns:
            TFDS builder instance.
        """
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, builder) -> dl.DLataset | tf.data.Dataset:
        """Build the dataset from the TFDS builder.

        Args:
            builder: TFDS builder instance.

        Returns:
            The constructed dataset.
        """
        raise NotImplementedError

    def get_dataset_name(self) -> str:
        """Get the dataset name.

        Override in subclasses if the name differs from self.dataset_name.
        """
        return self.dataset_name

    def _get_dataset_ops(self) -> tf.data.Options:
        """Get optimized tf.data.Options using the mixin."""
        want_full_determinism = self.want_full_determinism or self.want_val
        return self.get_dataset_ops(want_full_determinism)

    def apply_standalone_batching(self):
        """Apply batching and augmentation for standalone use.

        Call this at the end of __init__ if self.standalone is True.
        """
        if not self._standalone:
            return

        self.dataset = prepare_batched_dataset(
            dataset=self.dataset,
            want_val=self.want_val,
            shuffle=self._shuffle,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
            seed=self.seed,
            max_samples=self._max_samples,
            batch_size=self._batch_size,
            resize_resolution=self.config.resize_resolution,
            primary_image_key=self.spec.primary_image_key,
            wrist_image_key=self.spec.wrist_image_key,
            wrist_image_right_key=self.spec.wrist_image_right_key,
            aggressive_aug=getattr(self.config, "aggressive_aug", False),
            aug_wrist_image=getattr(self.config, "aug_wrist_image", True),
            not_rotate_wrist_prob=getattr(self.config, "not_rotate_wrist_prob", 0.0),
        )

    def __iter__(self):
        """Iterate over the dataset (requires standalone=True)."""
        assert self._standalone, "This dataset is not standalone"
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self) -> int:
        """Return the number of transitions in the dataset."""
        if self.dataset_statistics is None:
            return 0
        return self.dataset_statistics["state"].num_transitions


class BaseRobotDataset(BaseDataset, ValidationSplitMixin, RLDSDatasetMixin, TrajectoryIdentifierMixin):
    """Base class for single RLDS robot datasets.

    This class provides the common functionality for all robot datasets:
    - RLDS dataset loading with proper sharding
    - Train/val splitting with consistent hashing
    - Action chunking and state transformations
    - Statistics computation and caching

    Subclasses must implement:
    - get_traj_identifier(): Add trajectory IDs
    - apply_restructure(): Restructure raw data to standardized format
    - apply_traj_filters(): Apply trajectory-level filters
    - apply_frame_filters(): Apply frame-level filters
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: DataConfig,
        action_dim: int = 32,
        action_horizon: int = 16,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE
        seed: int = 0,
        split: str = "train",
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
        state_dim: int = 10,
    ):
        # Initialize base class
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            config=config,
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=action_horizon,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
            standalone=standalone,
            shuffle=shuffle,
            batch_size=batch_size,
            max_samples=max_samples,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )

        # Robot-specific settings
        self.use_wrist_image = bool(config.use_wrist_image)
        self.enable_prediction_training = enable_prediction_training
        self.pred_prob = pred_prob
        self.primary_pred_prob = primary_pred_prob
        self.horizon_seconds = config.horizon_seconds

        # Load dataset kwargs from central config
        dataset_kwargs = load_dataset_kwargs(
            dataset_name, data_dir, load_camera_views=("primary", "wrist", "wrist_right")
        )

        logging.info(f"Dataset kwargs: {dataset_kwargs}")
        self.control_frequency: int = int(dataset_kwargs["control_frequency"])
        self.standardize_fn = dataset_kwargs["standardize_fn"]
        self.image_obs_keys = dataset_kwargs["image_obs_keys"]
        self.state_obs_keys = dataset_kwargs["state_obs_keys"]
        self.state_encoding = dataset_kwargs["state_encoding"]
        self.action_encoding = dataset_kwargs["action_encoding"]
        self.is_bimanual = dataset_kwargs.get("is_bimanual", False)
        self.has_wrist_image = dataset_kwargs["image_obs_keys"]["wrist"] is not None

        # Build dataset
        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Check if we have cached statistics
        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)

        # If no cached stats, compute them first
        if cached_stats is None or self.config.force_recompute_stats:
            logging.info(f"No cached statistics found for {dataset_name} or force recompute. Computing statistics...")
            # Build temporary dataset for stats computation
            self.dataset = self.build_dataset(self.builder)
            self.get_traj_identifier()
            self.apply_restructure()
            self.apply_traj_transforms(action_horizon=action_horizon)

            # Compute and save statistics
            cached_stats = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="actions",
                state_key="state",
            )
            logging.info(f"Statistics computed and saved for {dataset_name}")

        # Now rebuild dataset using cached stats path for consistent ordering
        self.dataset = self.build_dataset(self.builder)
        self.get_traj_identifier()

        # Set statistics before filtering (needed for dataset-specific filters)
        self.dataset_statistics = cached_stats

        # Apply operations in consistent order: filter -> split -> restructure
        self.apply_traj_filters(action_key="action")
        self.split_val(split_seed=seed)
        self.apply_restructure()

        # If state encoding is NONE, ensure state stats are properly padded
        if self.state_encoding == StateEncoding.NONE:
            from lap.shared.normalize_adapter import ExtendedNormStats

            if len(self.dataset_statistics["state"].mean) == 0:
                self.dataset_statistics["state"] = ExtendedNormStats(
                    mean=np.zeros(self.state_dim, dtype=np.float32),
                    std=np.ones(self.state_dim, dtype=np.float32),
                    q01=np.zeros(self.state_dim, dtype=np.float32),
                    q99=np.zeros(self.state_dim, dtype=np.float32),
                    num_transitions=self.dataset_statistics["state"].num_transitions,
                    num_trajectories=self.dataset_statistics["state"].num_trajectories,
                )

        self.apply_traj_transforms(action_horizon=action_horizon)

        self.apply_repack_transforms()

        self.apply_flatten()

        self.apply_prediction_frame_transform()

        self.apply_frame_filters()

        # Apply standalone batching if needed
        self.apply_standalone_batching()

    def build_dataset_builder(self, ds_name, data_dir):
        """Build TFDS builder with version from config if available."""
        ds_name_with_version = get_tfds_name_with_version(ds_name)
        return tfds.builder(ds_name_with_version, data_dir=data_dir)

    def build_dataset(self, builder):
        """Build dataset from RLDS builder with proper sharding."""
        want_full_determinism = self.want_full_determinism or self.want_val

        # Use mixin for RLDS dataset building
        dataset = self.build_rlds_dataset(
            builder,
            shuffle=not want_full_determinism,
            num_parallel_reads=self.num_parallel_reads,
            shard_for_training=not self.want_val,
        )

        # Apply optimized options
        dataset = dataset.with_options(self._get_dataset_ops())
        return dataset

    def split_val(self, split_seed):
        def _split_filter(traj):
            salt = tf.strings.as_string(split_seed)
            anchor = traj["trajectory_id"][0]
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def chunk_actions(self, traj, action_horizon: int, action_key: str = "actions"):
        """Splits episode into action chunks with proper padding based on control mode.

        Control modes:
        - ABS_EEF_POS: Absolute end-effector poses -> use last-value padding + compute differences
        - EEF_POS, EEF_R6: Delta end-effector actions -> use zero padding, no difference computation
        - JOINT_POS, JOINT_POS_BIMANUAL: Joint position control -> use last-value padding
        """
        from lap.datasets.utils.helpers import ActionEncoding

        traj_len = tf.shape(traj[action_key])[0]
        action_encoding = self.action_encoding

        is_joint_pos = action_encoding in (ActionEncoding.JOINT_POS, ActionEncoding.JOINT_POS_BIMANUAL)

        if is_joint_pos:
            # Joint position control: use last-value padding, no difference computation
            traj[action_key] = gather_with_last_value_padding(
                data=traj[action_key],
                sequence_length=traj_len,
                window_size=action_horizon,
            )
        else:
            traj[action_key] = gather_with_last_value_padding(
                data=traj[action_key],
                sequence_length=traj_len,
                window_size=action_horizon + 1,
            )
            traj[action_key] = tf.concat(
                (
                    traj[action_key][:, 1:, :3] - traj[action_key][:, 0:1, :3],
                    euler_diff(
                        traj[action_key][:, 1:, 3:6],
                        traj[action_key][:, 0:1, 3:6],
                    ),
                    traj[action_key][:, :-1, 6:7],
                ),
                axis=-1,
            )

        return traj

    def apply_traj_transforms(
        self,
        action_horizon,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """Apply trajectory-level transforms (state conversion, action chunking, etc.)."""

        def state_euler_to_rot6d(traj):
            traj["observation"][state_key] = tf.concat(
                [
                    traj["observation"][state_key][:, :3],
                    euler_to_rot6d(traj["observation"][state_key][:, 3:6]),
                    traj["observation"][state_key][:, 6:],
                ],
                axis=-1,
            )
            traj["raw_state"] = tf.concat(
                [
                    traj["raw_state"][:, :3],
                    euler_to_rot6d(traj["raw_state"][:, 3:6]),
                    traj["raw_state"][:, 6:],
                ],
                axis=-1,
            )
            return traj

        self.dataset = self.dataset.traj_map(state_euler_to_rot6d, self.num_parallel_calls)

        self.dataset = self.dataset.traj_map(
            lambda traj: self.chunk_actions(traj, action_horizon=action_horizon, action_key=action_key),
            self.num_parallel_calls,
        )

        def pad_action_state(traj):
            # Pad actions to action_dim
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(
                traj[action_key],
                [[0, 0], [0, 0], [0, pad_amount_action]],
            )
            traj[action_key].set_shape([None, action_horizon, self.action_dim])

            # Pad state to state_dim
            state_last_dim = tf.shape(traj["observation"][state_key])[-1]
            pad_amount_state = tf.maximum(0, self.state_dim - state_last_dim)
            traj["observation"][state_key] = tf.pad(
                traj["observation"][state_key],
                [[0, 0], [0, pad_amount_state]],
            )
            traj["observation"][state_key].set_shape([None, self.state_dim])

            # Pad raw_state to state_dim
            raw_state_last_dim = tf.shape(traj["raw_state"])[-1]
            pad_amount_raw_state = tf.maximum(0, self.state_dim - raw_state_last_dim)
            traj["raw_state"] = tf.pad(
                traj["raw_state"],
                [[0, 0], [0, pad_amount_raw_state]],
            )
            traj["raw_state"].set_shape([None, self.state_dim])
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def group_language_actions(traj):
            """Compute per-timestep summed language actions over variable horizons."""
            traj_len = tf.shape(traj[action_key])[0]
            timestep_ids = tf.range(traj_len, dtype=tf.int32)
            remaining = tf.maximum(traj_len - timestep_ids, 1)

            horizon_seconds = tf.constant(self.horizon_seconds, dtype=tf.float32)
            control_freq = tf.cast(self.control_frequency, tf.float32)
            horizon_steps = tf.cast(tf.round(horizon_seconds * control_freq), tf.int32)
            horizon_steps = tf.maximum(horizon_steps, 1)

            traj_id_hash = tf.strings.to_hash_bucket_fast(traj["trajectory_id"][0], 2147483647)
            seed_pair = [self.seed, traj_id_hash]
            horizon_idx = tf.random.stateless_uniform(
                [traj_len],
                seed=seed_pair,
                minval=0,
                maxval=tf.shape(horizon_steps)[0],
                dtype=tf.int32,
            )
            chosen_steps = tf.gather(horizon_steps, horizon_idx)
            valid_lengths = tf.minimum(chosen_steps, remaining)

            max_window = tf.reduce_max(horizon_steps)
            actions_window = gather_with_padding(
                data=traj["language_action"],
                sequence_length=traj_len,
                window_size=max_window,
                per_timestep_windows=chosen_steps,
            )

            traj["language_actions"] = sum_actions(actions_window, valid_lengths)
            traj["language_actions"].set_shape([None, traj["language_action"].shape[-1]])

            actual_horizon_seconds = tf.cast(valid_lengths, tf.float32) / control_freq
            traj["time_horizon_seconds"] = actual_horizon_seconds
            traj["time_horizon_seconds"].set_shape([None])

            return traj

        self.dataset = self.dataset.traj_map(group_language_actions, self.num_parallel_calls)

        def add_prediction_pairs(traj):
            """Add prediction frame pairs and corresponding language actions."""
            if not self.enable_prediction_training:
                return traj
            traj_len = tf.shape(traj[action_key])[0]

            max_horizon = int(2.5 * self.control_frequency)
            max_horizon_clamped = tf.minimum(max_horizon, traj_len - 1)
            max_horizon_clamped = tf.maximum(max_horizon_clamped, 1)

            traj_id_hash = tf.strings.to_hash_bucket_fast(traj["trajectory_id"][0], 2147483647)
            seed_pair = [self.seed, traj_id_hash]

            deltas = tf.random.stateless_uniform(
                [traj_len],
                seed=seed_pair,
                minval=max_horizon_clamped,
                maxval=max_horizon_clamped + 1,
                dtype=tf.int32,
            )
            future_indices = tf.minimum(tf.range(traj_len, dtype=tf.int32) + deltas, traj_len - 1)

            current_imgs = traj["observation"][self.spec.primary_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.primary_image_key] = tf.stack([current_imgs, future_imgs], axis=1)

            current_imgs = traj["observation"][self.spec.wrist_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.wrist_image_key] = tf.stack([current_imgs, future_imgs], axis=1)

            actions_window = gather_with_padding(
                data=traj["language_action"],
                sequence_length=traj_len,
                window_size=max_horizon,
                per_timestep_windows=deltas,
            )

            prediction_lang_actions = sum_actions(actions_window, deltas)
            prediction_lang_actions.set_shape([None, traj["language_action"].shape[-1]])

            traj["prediction_language_actions"] = prediction_lang_actions
            traj["prediction_delta"] = deltas

            return traj

        self.dataset = self.dataset.traj_map(add_prediction_pairs, self.num_parallel_calls)

    def apply_repack_transforms(self):
        """Repack trajectory for output format."""

        def common(traj):
            traj_len = tf.shape(traj["actions"])[0]
            traj["caption"] = tf.repeat(tf.constant("", dtype=tf.string), traj_len)
            traj["is_vqa_sample"] = tf.repeat(tf.constant(False, dtype=tf.bool), traj_len)
            traj["vqa_dataset_id"] = tf.repeat(tf.constant(0, dtype=tf.int32), traj_len)
            return traj

        self.dataset = self.dataset.traj_map(common, self.num_parallel_calls)

    def get_traj_identifier(self):
        raise NotImplementedError

    def apply_restructure(self):
        raise NotImplementedError

    def apply_traj_filters(self, action_key):
        raise NotImplementedError

    def apply_frame_filters(self):
        raise NotImplementedError

    def apply_flatten(self):
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_prediction_frame_transform(self):
        """Apply prediction frame transformation after flattening."""
        if not self.enable_prediction_training:

            def add_prediction_mask(sample):
                sample["is_prediction_sample"] = tf.constant(False, dtype=tf.bool)
                sample["pred_use_primary"] = tf.constant(False, dtype=tf.bool)
                sample.pop("trajectory_id")
                return sample

            self.dataset = self.dataset.frame_map(add_prediction_mask, num_parallel_calls=self.num_parallel_calls)
            return

        def convert_to_prediction_sample(sample):
            """Randomly convert samples to prediction samples based on pred_prob."""
            traj_id_str = sample["trajectory_id"]
            traj_id_str = tf.reshape(traj_id_str, [])
            traj_id_hash = tf.strings.to_hash_bucket_fast(traj_id_str, 2147483647)
            frame_id_str = tf.strings.join([traj_id_str, "_frame"])
            frame_hash = tf.cast(tf.strings.to_hash_bucket_fast(frame_id_str, 2147483647), tf.int64)
            seed_pair = [self.seed + traj_id_hash, frame_hash]

            is_pred_sample = tf.random.stateless_uniform([], seed=seed_pair) < self.pred_prob

            wrist_sample = sample["observation"][self.spec.wrist_image_key][0]
            has_wrist_image = tf.greater(tf.strings.length(wrist_sample), 0)

            use_primary = tf.cond(
                has_wrist_image,
                lambda: tf.random.stateless_uniform([], seed=[seed_pair[0] + 1, seed_pair[1]]) < self.primary_pred_prob,
                lambda: tf.constant(True, dtype=tf.bool),
            )

            primary_frame0 = sample["observation"][self.spec.primary_image_key][0]
            primary_frame1 = sample["observation"][self.spec.primary_image_key][1]
            wrist_frame0 = sample["observation"][self.spec.wrist_image_key][0]
            wrist_frame1 = sample["observation"][self.spec.wrist_image_key][1]

            def use_primary_camera():
                return primary_frame0, primary_frame1

            def use_wrist_camera():
                return wrist_frame0, wrist_frame1

            pred_primary_img, pred_wrist_img = tf.cond(use_primary, use_primary_camera, use_wrist_camera)

            sample["pred_use_primary"] = use_primary

            normal_primary_img = primary_frame0
            normal_wrist_img = wrist_frame0

            final_primary_img = tf.cond(is_pred_sample, lambda: pred_primary_img, lambda: normal_primary_img)
            final_wrist_img = tf.cond(is_pred_sample, lambda: pred_wrist_img, lambda: normal_wrist_img)

            sample["observation"][self.spec.primary_image_key] = final_primary_img
            sample["observation"][self.spec.wrist_image_key] = final_wrist_img

            pred_horizon_seconds = None
            if "prediction_delta" in sample:
                pred_horizon_seconds = tf.cast(sample["prediction_delta"], tf.float32) / tf.cast(
                    self.control_frequency, tf.float32
                )

            if "prediction_language_actions" in sample and "language_actions" in sample:
                sample["language_actions"] = tf.cond(
                    is_pred_sample,
                    lambda: sample["prediction_language_actions"],
                    lambda: sample["language_actions"],
                )

            if "time_horizon_seconds" in sample and "prediction_delta" in sample:
                if pred_horizon_seconds is None:
                    pred_horizon_seconds = tf.cast(sample["prediction_delta"], tf.float32) / tf.cast(
                        self.control_frequency, tf.float32
                    )
                sample["time_horizon_seconds"] = tf.cond(
                    is_pred_sample,
                    lambda: pred_horizon_seconds,
                    lambda: sample["time_horizon_seconds"],
                )

            sample["is_prediction_sample"] = is_pred_sample
            sample.pop("trajectory_id")

            sample.pop("prediction_language_actions", None)
            sample.pop("prediction_delta", None)

            return sample

        self.dataset = self.dataset.frame_map(convert_to_prediction_sample, self.num_parallel_calls)


def _matrix_to_euler_xyz_extrinsic(R: tf.Tensor) -> tf.Tensor:
    """Convert rotation matrix back to extrinsic XYZ Euler angles."""
    sy = tf.sqrt(tf.maximum(R[..., 0, 0] * R[..., 0, 0] + R[..., 1, 0] * R[..., 1, 0], 1e-12))
    singular = sy < 1e-6

    roll = tf.where(
        singular,
        tf.atan2(-R[..., 1, 2], R[..., 1, 1]),
        tf.atan2(R[..., 2, 1], R[..., 2, 2]),
    )
    pitch = tf.atan2(-R[..., 2, 0], sy)
    yaw = tf.where(
        singular,
        tf.zeros_like(R[..., 0, 0]),
        tf.atan2(R[..., 1, 0], R[..., 0, 0]),
    )

    return tf.stack([roll, pitch, yaw], axis=-1)


def sum_actions(actions: tf.Tensor, valid_lengths: tf.Tensor | None = None) -> tf.Tensor:
    """Sum sequences of actions (x, y, z, roll, pitch, yaw).

    Args:
        actions: Tensor with shape [T, W, A] where the last dimension starts with
            (x, y, z, roll, pitch, yaw). Roll, pitch, yaw use extrinsic XYZ order.
        valid_lengths: Optional [T] tensor specifying how many valid deltas to use per window.

    Returns:
        Tensor of shape [T, A] where each row encodes the summed delta over the window.
    """
    actions = tf.convert_to_tensor(actions)
    num_windows = tf.shape(actions)[0]
    window_size = tf.shape(actions)[1]

    if valid_lengths is None:
        valid_lengths = tf.fill([num_windows], window_size)
    else:
        valid_lengths = tf.cast(valid_lengths, tf.int32)

    valid_lengths = tf.minimum(valid_lengths, window_size)
    valid_lengths = tf.maximum(valid_lengths, 1)

    def _sum_single_window(inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        window, length = inputs
        length = tf.cast(length, tf.int32)
        window = window[:length]

        action_dim = tf.shape(window)[-1]
        pad_amt = tf.maximum(0, 6 - action_dim)
        window = tf.pad(window, [[0, 0], [0, pad_amt]])
        padded_dim = tf.shape(window)[-1]

        translation_sum = tf.reduce_sum(window[:, :3], axis=0)

        def _compose_rotation(R_total, rpy):
            R_step = _R_from_euler_xyz(rpy)
            return tf.linalg.matmul(R_total, R_step)

        R_final = tf.foldl(
            _compose_rotation,
            window[:, 3:6],
            initializer=tf.eye(3, dtype=window.dtype),
        )
        rpy_final = _matrix_to_euler_xyz_extrinsic(R_final)

        last_index = tf.maximum(length - 1, 0)
        last_action = tf.gather(window, last_index)
        tail_indices = tf.range(6, padded_dim)
        tail = tf.gather(last_action, tail_indices)

        summed = tf.concat([translation_sum, rpy_final, tail], axis=0)
        return summed

    output_spec = tf.TensorSpec(shape=actions.shape[-1:], dtype=actions.dtype)
    return tf.map_fn(_sum_single_window, (actions, valid_lengths), fn_output_signature=output_spec)
