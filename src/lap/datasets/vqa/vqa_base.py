"""Base class and registry for VQA datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from dlimp import DLataset
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.base_dataset import BaseDataset
from lap.datasets.output_schema import ObservationBuilder
from lap.datasets.output_schema import TrajectoryOutputBuilder
from lap.datasets.registry import get_num_vqa_datasets
from lap.datasets.registry import get_vqa_dataset_id
from lap.datasets.registry import get_vqa_dataset_name
from lap.datasets.utils.helpers import ActionEncoding
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import StateEncoding

if TYPE_CHECKING:
    from lap.training.config import DataConfig


# Number of VQA dataset types (for metrics computation)
# Use function for dynamic count based on registered datasets
def get_num_vqa_dataset_types() -> int:
    """Get number of VQA dataset types (for metrics computation)."""
    return get_num_vqa_datasets()


# Snapshot alias.
NUM_VQA_DATASETS: int = get_num_vqa_datasets()


# Reverse mapping for decoding - use registry function
def get_vqa_name_by_id(dataset_id: int) -> str | None:
    """Get VQA dataset name by ID."""
    return get_vqa_dataset_name(dataset_id)


def ensure_dldataset(ds, is_flattened=False):
    """Ensure dataset is a DLataset instance."""
    if isinstance(ds, tf.data.Dataset) and not isinstance(ds, DLataset):
        Original = ds.__class__
        MixedDL = type("DLatasetWrapped", (DLataset, Original), {})
        ds.__class__ = MixedDL
        ds.is_flattened = is_flattened
    return ds


class BaseVQADataset(BaseDataset):
    """Base class for VQA datasets (COCO Captions, VQAv2, etc.).

    This class inherits from BaseDataset and adds VQA-specific functionality:
    - No language actions in the traditional sense
    - Single frame (no temporal structure)
    - No proprioceptive state
    - Dummy actions for compatibility
    - Caption-based outputs

    Subclasses must implement:
    - build_dataset_builder(): Return TFDS builder
    - get_dataset_name(): Return the dataset name string
    - get_num_transitions(): Return approximate number of samples for statistics
    - create_trajectory_id(): Create trajectory ID from example
    - extract_prompt_and_caption(): Extract prompt and caption from example
    - extract_and_encode_image(): Extract and encode image from example
    """

    # Dataset-specific name (to be set by subclasses)
    DATASET_NAME: ClassVar[str] = ""

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: DataConfig,
        action_horizon: int = 16,
        action_dim: int = 32,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        seed: int = 0,
        split: str = "train",
        standalone: bool = True,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        enable_prediction_training: bool = False,
        pred_prob: float | None = None,
        primary_pred_prob: float | None = None,
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

        # VQA-specific settings (shared across all VQA datasets)
        self.use_wrist_image = False  # VQA has no wrist images
        self.control_frequency = 1  # Single frame, no temporal control
        self.image_obs_keys = {"primary": None}
        self.state_obs_keys = []
        self.state_encoding = StateEncoding.NONE
        self.action_encoding = ActionEncoding.EEF_POS
        self.is_bimanual = False

        # Build TFDS dataset (subclass-specific)
        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Build dataset (subclass-specific)
        self.dataset = self.build_dataset(self.builder, split)

        # Split train/val
        if split in ["train", "val"]:
            self.split_val(split_seed=seed)

        self.apply_vqa_restructure()
        self.apply_vqa_transforms()
        self.apply_vqa_frame_filters()

        # Create dummy statistics for compatibility using mixin
        num_transitions = self.get_num_transitions()
        self.dataset_statistics = self.build_dummy_statistics(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            num_transitions=num_transitions,
        )

        # Apply standalone batching
        self.apply_standalone_batching()

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement build_dataset_builder")

    def build_dataset(self, builder, split: str = "train"):
        """Build TensorFlow dataset from TFDS builder."""
        want_full_determinism = self.want_full_determinism or self.want_val

        read_config = tfds.ReadConfig(
            shuffle_seed=self.seed,
        )

        ds = builder.as_dataset(
            split="train",
            shuffle_files=bool(not want_full_determinism),
            read_config=read_config,
        )
        ds = ensure_dldataset(ds, is_flattened=True)

        dataset = ds.shard(jax.process_count(), jax.process_index())
        dataset = dataset.with_options(self._get_dataset_ops())

        return dataset

    def get_num_transitions(self) -> int:
        """Get approximate number of transitions for statistics. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement get_num_transitions")

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from example. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_trajectory_id")

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from example. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement extract_prompt_and_caption")

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode image from example. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement extract_and_encode_image")

    def split_val(self, split_seed: int):
        """Split dataset into train/val using consistent hashing."""

        def _split_filter(example):
            salt = tf.strings.as_string(split_seed)
            trajectory_id = self.create_trajectory_id(example)
            key = tf.strings.join([salt, trajectory_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def apply_vqa_restructure(self):
        """Restructure VQA data to match robot dataset format using output schema builder."""

        def restructure(example):
            image_encoded = self.extract_and_encode_image(example)
            prompt, caption = self.extract_prompt_and_caption(example)

            dataset_name_str = self.get_dataset_name()
            # Use registry's auto-assigned VQA dataset ID
            vqa_dataset_id = get_vqa_dataset_id(dataset_name_str)

            # Use output schema builder for consistent structure
            observation = ObservationBuilder.build_vqa_observation(
                image_encoded=image_encoded,
                primary_image_key=self.spec.primary_image_key,
                wrist_image_key=self.spec.wrist_image_key,
                state_dim=self.state_dim,
                wrist_image_right_key=self.spec.wrist_image_right_key,
            )

            return TrajectoryOutputBuilder.build_vqa_frame(
                observation=observation,
                prompt=prompt,
                caption=caption,
                dataset_name=dataset_name_str,
                vqa_dataset_id=vqa_dataset_id,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
            )

        self.dataset = self.dataset.frame_map(
            restructure,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_transforms(self):
        """Apply VQA-specific transforms (shared across all VQA datasets)."""

        def transform(example):
            # Ensure actions and language_actions are properly shaped
            example["actions"] = tf.zeros(
                [self.action_horizon, self.action_dim],
                dtype=tf.float32,
            )
            example["language_actions"] = tf.zeros([7], dtype=tf.float32)
            return example

        self.dataset = self.dataset.frame_map(
            transform,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_frame_filters(self):
        """Filter out samples with empty questions or captions."""

        def has_valid_qa(example):
            has_question = tf.strings.length(example["prompt"]) > 0
            has_answer = tf.strings.length(example["caption"]) > 0
            return tf.logical_and(has_question, has_answer)

        self.dataset = self.dataset.filter(has_valid_qa)
