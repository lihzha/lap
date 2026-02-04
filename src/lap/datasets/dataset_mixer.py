"""Multi-dataset orchestration for RLDS datasets."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import numpy as np
import tensorflow as tf

from lap.datasets.registry import VQA_DATASET_NAMES
from lap.datasets.registry import get_dataset_class
from lap.datasets.registry import is_vqa_dataset
from lap.datasets.robot.oxe_datasets import SingleOXEDataset
from lap.datasets.utils.mixtures import OXE_NAMED_MIXTURES

# Auto-discover and register datasets lazily at runtime to avoid import-time side effects
from lap.datasets.utils.dataset_discovery import ensure_datasets_registered
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import state_encoding_to_type
from lap.datasets.utils.normalization_and_config import allocate_threads
from lap.datasets.utils.normalization_and_config import pprint_data_mixture
from lap.datasets.utils.specs import RldsDatasetSpec
from lap.datasets.utils.statistics import GlobalStatisticsBuilder
from lap.datasets.utils.tfdata_pipeline import prepare_batched_dataset
from lap.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from lap.training.config import DataConfig


class OXEDatasets:
    """Multi-dataset mixer for OXE and VQA datasets.

    This class orchestrates loading and mixing multiple datasets with:
    - Automatic dataset class dispatch via registry
    - Global normalization across datasets
    - Weighted sampling for balanced training

    The refactored version uses:
    - Auto-discovery of datasets (no explicit imports needed)
    - GlobalStatisticsBuilder for cleaner stats computation
    - Simplified dataset instantiation via registry
    """

    spec: ClassVar[RldsDatasetSpec] = RldsDatasetSpec()

    def __init__(
        self,
        config: DataConfig,
        data_dir: str,
        action_dim: int = 32,
        state_dim: int = 10,
        action_horizon: int = 16,
        seed: int = 0,
        split: str = "train",
        *,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        hash_tables: dict | None = None,
        standalone: bool = True,
        use_global_normalization: bool = True,
        enable_prediction_training: bool = False,
    ):
        # Register dataset classes at construction time, not import time.
        ensure_datasets_registered()

        self.hash_tables = hash_tables
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        balance_weights = config.balance_weights

        # Set global seed for file-level operations
        tf.random.set_seed(seed)

        # Configure RLDS Dataset(s)
        assert config.data_mix in OXE_NAMED_MIXTURES, f"Unknown data mix: {config.data_mix}"
        mixture_spec = OXE_NAMED_MIXTURES[config.data_mix]
        self.config = config

        dataset_names = [it[0] for it in mixture_spec]
        sample_weights = [it[1] for it in mixture_spec]

        want_val = split == "val"

        # Allocate threads based on weights
        total_threads = len(os.sched_getaffinity(0))
        total_read_threads = int(total_threads * 0.4)
        total_transform_threads = int(total_threads * 0.4)
        logging.info(f"Total read threads: {total_read_threads}")
        logging.info(f"Total transform threads: {total_transform_threads}")

        threads_per_dataset = allocate_threads(total_transform_threads, np.array(sample_weights))
        reads_per_dataset = allocate_threads(total_read_threads, np.array(sample_weights))

        logging.info("Threads per Dataset: %s", threads_per_dataset)
        logging.info("Reads per Dataset: %s", reads_per_dataset)

        datasets, dataset_sizes, all_dataset_statistics = [], [], {}
        dataset_state_encodings = {}
        has_robot_dataset = False

        logging.info("Constructing datasets...")
        for dataset_name, threads, reads in zip(
            dataset_names,
            threads_per_dataset,
            reads_per_dataset,
        ):
            assert threads != tf.data.AUTOTUNE, "threads should not be AUTOTUNE"
            assert reads != tf.data.AUTOTUNE, "reads should not be AUTOTUNE"

            # Build common kwargs for all datasets
            kwargs = self._build_dataset_kwargs(
                dataset_name=dataset_name,
                data_dir=data_dir,
                config=config,
                action_horizon=action_horizon,
                action_dim=action_dim,
                state_dim=state_dim,
                seed=seed,
                split=split,
                action_proprio_normalization_type=action_proprio_normalization_type,
                threads=threads,
                reads=reads,
                enable_prediction_training=enable_prediction_training,
            )

            # Get dataset class from registry and instantiate
            ds = self._instantiate_dataset(dataset_name, kwargs)

            # Track whether this is a robot dataset (affects normalization)
            if not is_vqa_dataset(dataset_name):
                has_robot_dataset = True

            datasets.append(ds.dataset)
            dataset_statistics = ds.dataset_statistics
            dataset_sizes.append(dataset_statistics["state"].num_transitions)
            all_dataset_statistics[dataset_name] = dataset_statistics
            dataset_state_encodings[dataset_name] = config.state_encoding

        # Calculate sampling weights
        primary_dataset_indices = np.array(list(range(len(sample_weights))))

        if balance_weights:
            sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
        unnormalized_sample_weights = sample_weights.copy()
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        # Effective dataset length
        dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

        self.sample_weights = sample_weights
        self.unnormalized_sample_weights = unnormalized_sample_weights
        self.dataset_statistics = all_dataset_statistics
        self.dataset_length = dataset_len

        pprint_data_mixture(dataset_names, sample_weights)

        # Apply global normalization if requested
        logging.info(
            f"Global normalization check: use_global_normalization={use_global_normalization}, "
            f"has_robot_dataset={has_robot_dataset}, split={split} "
        )

        if use_global_normalization and has_robot_dataset:
            # Use extracted GlobalStatisticsBuilder
            stats_builder = GlobalStatisticsBuilder(action_dim=action_dim, state_dim=state_dim)
            global_stats = stats_builder.compute_global_stats(
                all_dataset_statistics=all_dataset_statistics,
                dataset_state_encodings=dataset_state_encodings,
                vqa_dataset_names=VQA_DATASET_NAMES,
            )

            logging.info(
                "Applying global normalization with stats with normalization type %s: %s",
                action_proprio_normalization_type,
                global_stats,
            )

            # Log statistics
            action_stats = global_stats["actions"]
            GlobalStatisticsBuilder.log_global_stats(
                "Action",
                action_dim,
                action_stats.min,
                action_stats.max,
                action_stats.q01,
                action_stats.q99,
                action_stats.mean,
                action_stats.std,
            )

            for state_type in ["joint_pos", "eef_pose"]:
                state_key = f"state_{state_type}"
                if state_key in global_stats:
                    state_stats = global_stats[state_key]
                    GlobalStatisticsBuilder.log_global_stats(
                        f"State ({state_type})",
                        state_dim,
                        state_stats.min,
                        state_stats.max,
                        state_stats.q01,
                        state_stats.q99,
                        state_stats.mean,
                        state_stats.std,
                    )

            # Apply state-type-specific normalization to each dataset BEFORE interleaving
            normalized_datasets = self._apply_normalization(
                datasets=datasets,
                dataset_names=dataset_names,
                dataset_state_encodings=dataset_state_encodings,
                global_stats=global_stats,
                action_proprio_normalization_type=action_proprio_normalization_type,
            )

            repeated_datasets = [ds.repeat() for ds in normalized_datasets] if not want_val else normalized_datasets

            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
                repeated_datasets, self.sample_weights, rerandomize_each_iteration=not want_val, seed=seed
            )
            self.global_statistics = global_stats
        else:
            repeated_datasets = [ds.repeat() for ds in datasets] if not want_val else datasets
            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
                repeated_datasets, self.sample_weights, rerandomize_each_iteration=not want_val, seed=seed
            )
            self.global_statistics = None

        self.dataset = prepare_batched_dataset(
            dataset=self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
            batch_size=batch_size,
            resize_resolution=config.resize_resolution,
            primary_image_key=self.spec.primary_image_key,
            wrist_image_key=self.spec.wrist_image_key,
            wrist_image_right_key=self.spec.wrist_image_right_key,
            aggressive_aug=getattr(config, "aggressive_aug", False),
            aug_wrist_image=getattr(config, "aug_wrist_image", True),
            not_rotate_wrist_prob=getattr(config, "not_rotate_wrist_prob", 0.0),
        )

    def _build_dataset_kwargs(
        self,
        dataset_name: str,
        data_dir: str,
        config: DataConfig,
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        seed: int,
        split: str,
        action_proprio_normalization_type: NormalizationType,
        threads: int,
        reads: int,
        enable_prediction_training: bool,
    ) -> dict:
        """Build common kwargs for dataset instantiation."""
        return {
            "data_dir": data_dir,
            "config": config,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "seed": seed,
            "split": split,
            "action_proprio_normalization_type": action_proprio_normalization_type,
            "num_parallel_reads": threads,
            "num_parallel_calls": threads,
            "standalone": False,
            "enable_prediction_training": enable_prediction_training,
            "pred_prob": config.pred_prob,
            "primary_pred_prob": config.primary_pred_prob,
            "state_dim": state_dim,
        }

    def _instantiate_dataset(self, dataset_name: str, kwargs: dict):
        """Instantiate dataset using registry dispatch.

        Uses the registry to find the appropriate dataset class,
        falling back to SingleOXEDataset for unknown datasets.
        """
        # Get dataset class from registry
        dataset_cls = get_dataset_class(dataset_name)

        if dataset_cls is None:
            # Fall back to default OXE dataset
            dataset_cls = SingleOXEDataset
            kwargs["dataset_name"] = dataset_name

        # Check if dataset needs hash_tables (DROID-specific)
        if hasattr(dataset_cls, "REQUIRES_HASH_TABLES") and dataset_cls.REQUIRES_HASH_TABLES:
            kwargs["hash_tables"] = self.hash_tables

        # Add dataset_name for classes that need it
        # Import here to avoid circular imports
        from lap.datasets.robot.droid_dataset import DroidDataset

        if "dataset_name" not in kwargs and dataset_cls != DroidDataset:
            kwargs["dataset_name"] = dataset_name

        # Instantiate
        ds = dataset_cls(**kwargs)

        # Update hash tables if returned (DROID-specific)
        if hasattr(ds, "hash_tables") and ds.hash_tables:
            self.hash_tables = ds.hash_tables
        elif hasattr(ds, "ep_table"):
            # Support direct table access on datasets exposing table attributes
            self.hash_tables = {
                "ep_table": getattr(ds, "ep_table", None),
                "filter_table": getattr(ds, "filter_table", None),
                "instr_table": getattr(ds, "instr_table", None),
            }

        return ds

    def _apply_normalization(
        self,
        datasets: list,
        dataset_names: list[str],
        dataset_state_encodings: dict,
        global_stats: dict,
        action_proprio_normalization_type: NormalizationType,
    ) -> list:
        """Apply state-type-specific normalization to each dataset."""
        normalized_datasets = []

        for ds_name, ds in zip(dataset_names, datasets):
            if is_vqa_dataset(ds_name):
                # Skip normalization for VQA datasets
                normalized_datasets.append(ds)
                continue

            state_enc = dataset_state_encodings[ds_name]
            state_type = state_encoding_to_type(state_enc)

            # Create normalizer for this state type
            stats = {"actions": global_stats["actions"]}
            state_key_name = f"state_{state_type}"
            if state_key_name in global_stats:
                stats["state"] = global_stats[state_key_name]

            normalizer = NormalizeActionAndProprio(
                norm_stats=stats,
                normalization_type=action_proprio_normalization_type,
                action_key="actions",
                state_key="state",
            )

            normalized_datasets.append(ds.map(normalizer, num_parallel_calls=tf.data.AUTOTUNE))

        return normalized_datasets

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        return self.dataset_length

    @property
    def num_batches_per_epoch(self) -> int:
        """Compute number of batches to iterate over the full dataset for one epoch."""
        import jax

        per_step = self.batch_size * jax.process_count()
        if per_step <= 0:
            return 0
        return (self.dataset_length + per_step - 1) // per_step

    @property
    def num_val_batches_per_epoch(self) -> int:
        """Compute number of batches per epoch based on dataset length and batch size."""
        num_transitions = next(v.num_transitions for k, v in self.global_statistics.items() if "state" in k)

        return int(num_transitions * self.config.val_fraction * 0.8 // (self.batch_size))
