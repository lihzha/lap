# lap/training/data_loader.py
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Literal

import jax
import numpy as np
import openpi.models.model as _model
import openpi.training.data_loader as up  # upstream module
import openpi.transforms as up_tf
import tensorflow as tf

from lap.datasets.dataset_mixer import OXEDatasets
from lap.models.model_adapter import CoTObservation
from lap.models.tokenizer import PaligemmaTokenizer
import lap.training.config as _config


def _create_rlds_dataset(
    data_cfg: _config.DataConfig,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    *,
    enable_prediction_training: bool = False,
    shuffle: bool,
    seed: int,
    max_samples: int | None,
    split: str,
    hash_tables: dict | None = None,
) -> up.Dataset:
    # Per-host batching; avoids redundant slicing work in multi-process setups
    local_bsz = max(1, batch_size // jax.process_count())

    rlds_data_dir = getattr(data_cfg, "rlds_data_dir", None)
    assert rlds_data_dir is not None, "rlds_data_dir is required"
    dataset_cls = OXEDatasets

    # Build kwargs dynamically
    kwargs = {
        "data_dir": rlds_data_dir,
        "batch_size": local_bsz,
        "shuffle": shuffle,
        "max_samples": max_samples,
        "seed": seed,
        "config": data_cfg,
        "split": split,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "hash_tables": hash_tables,
        "standalone": True,
        "action_proprio_normalization_type": data_cfg.action_proprio_normalization_type,
        "enable_prediction_training": enable_prediction_training,
    }

    return dataset_cls(**kwargs)


def _make_iterable_transforms(
    data_cfg: _config.DataConfig,
    *,
    skip_norm_stats: bool,
    split: str | None,
) -> list[up_tf.DataTransformFn]:
    norm_stats = {}
    if data_cfg.repo_id != "fake" and not skip_norm_stats:
        if data_cfg.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_cfg.norm_stats

    if norm_stats is None:
        logging.info("Not using normalization stats in the data_loader.")

    tx = [
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        *data_cfg.model_transforms.inputs,
    ]  # normalize is handled on dataset level

    if split is not None and split != "train":
        new_tx = []
        for t in tx:
            if hasattr(t, "wrist_image_dropout_prob"):
                new_tx.append(dataclasses.replace(t, wrist_image_dropout_prob=0.0))
            else:
                new_tx.append(t)
        tx = new_tx

    return tx


class IterableTransformedDataset(up.IterableTransformedDataset):
    def __init__(self, batch_size, *args, persistent_iterator=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.persistent_iterator = persistent_iterator

    def __iter__(self):
        # Regular behavior: create new iterator each time
        # This already yields numpy arrays via as_numpy_iterator()
        dataset_iter = iter(self._dataset)

        for sample in dataset_iter:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(self.batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)


# ---------- Public entry: create_data_loader ----------
def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    seed: int = 0,
    max_samples: int | None = None,
    split: str = "train",
    framework: Literal["jax", "pytorch"] = "jax",
    hash_tables: dict | None = None,
    persistent_iterator: bool = False,
) -> up.DataLoader[tuple[CoTObservation, _model.Actions]]:
    # Avoid import-time side effects:
    # Only clear LEROBOT_HOME if we are about to construct a LeRobot dataset.
    if config.data.repo_id not in (None, "fake") and config.data.rlds_data_dir is None:
        os.environ.pop("LEROBOT_HOME", None)

    data_cfg: _config.DataConfig = config.data.create(config.assets_dirs, config.model)
    logging.info("data_config: %s", data_cfg)

    # If RLDS, follow the RLDS path with our two hooks; else, fall back to upstream torch loader
    if data_cfg.rlds_data_dir is not None:
        if framework == "pytorch":
            raise NotImplementedError("PyTorch RLDS data loader is not supported yet")

        # 1) dataset
        ds = _create_rlds_dataset(
            data_cfg=data_cfg,
            batch_size=config.batch_size,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            enable_prediction_training=config.model.enable_prediction_training,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples if max_samples is not None else getattr(data_cfg, "max_samples", None),
            split=split,
            hash_tables=hash_tables,
        )

        # 2) transforms (split-aware)
        tx = _make_iterable_transforms(data_cfg, skip_norm_stats=data_cfg.norm_stats is None, split=split)
        iterable = IterableTransformedDataset(
            max(1, config.batch_size // jax.process_count()),
            ds,
            tx,
            is_batched=True,
            persistent_iterator=persistent_iterator,
        )

        return RLDSDataLoader(
            iterable,
            sharding=sharding,
            num_batches=num_batches,
            data_cfg=data_cfg,
            persistent_iterator=persistent_iterator,
            split=split,
        )

    # Non-RLDS: delegate entirely to upstream (this will require torch if used)
    return up.create_torch_data_loader(
        data_cfg,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
        skip_norm_stats=data_cfg.norm_stats is None,
        framework=framework,
    )


class RLDSDataLoader:
    """Iterates an IterableTransformedDataset and returns sharded jax.Arrays.

    If you run on multiple JAX processes (e.g. multi-host TPU), each process
    automatically receives its 1/`process_count` share of every batch.
    """

    def __init__(
        self,
        dataset: up.IterableTransformedDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        data_cfg: _config.DataConfig,
        persistent_iterator: bool = False,
        split: str = "train",
    ):
        self._dataset = dataset
        self._original_dataset = dataset  # Keep reference for skip-based resumption
        self._num_batches = num_batches
        self._data_cfg = data_cfg
        self._n_proc = jax.process_count()
        self._proc_idx = jax.process_index()
        self._persistent_iterator = persistent_iterator
        self._iterator = None
        self._checkpoint = None
        self._seen_batches = 0
        self._skip_batches = 0  # Track how many batches to skip on next iteration
        self._split = split  # Track split to determine if we should stop on StopIteration

        if sharding is None:
            sharding = jax.sharding.PositionalSharding(jax.local_devices())
        self._sharding = sharding

    def _to_device(self, batch):
        def put(x):
            if not (hasattr(x, "shape") and x.shape):
                return x
            # Skip strings/bytes/object arrays – JAX can't put them on device
            if hasattr(x, "dtype") and (x.dtype == np.object_ or getattr(x.dtype, "kind", None) in ("U", "S")):
                return x
            if isinstance(self._sharding, jax.sharding.NamedSharding):
                return jax.make_array_from_process_local_data(self._sharding, x)
            return jax.device_put(x, self._sharding)

        return jax.tree_util.tree_map(put, batch)

    def _assert_divisible(self, batch):
        sizes = [x.shape[0] for x in jax.tree_util.tree_leaves(batch) if hasattr(x, "shape") and x.shape]
        if not sizes:
            return
        b = max(sizes)  # this is per-host if dataset was sharded

        if isinstance(self._sharding, jax.sharding.NamedSharding):
            mesh = self._sharding.mesh
            # DATA axis size across the whole mesh:
            data_axis_size = mesh.shape.get("data", None)  # or use your DATA_AXIS constant
            if data_axis_size is None:
                return  # no data axis; nothing to check

            # Special case: for cross-host FSDP when data_axis_size == 1,
            # we don't need data parallelism across hosts - each host gets the same data
            # and works together on FSDP sharding
            if data_axis_size == 1 and self._n_proc > 1:
                # Cross-host FSDP: validate against local device count instead
                ldc = jax.local_device_count()
                if b % ldc != 0:
                    raise ValueError(
                        f"Per-host batch {b} must be divisible by local_device_count {ldc} for cross-host FSDP"
                    )
                return

            # Standard data parallelism validation
            dp_per_host = data_axis_size // self._n_proc
            if dp_per_host == 0 or data_axis_size % self._n_proc != 0:
                raise ValueError("Mesh/data axis inconsistent with process_count.")
            if b % dp_per_host != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by dp_per_host {dp_per_host}")
        else:
            # PositionalSharding fallback shards leading axis across local devices
            ldc = jax.local_device_count()
            if b % ldc != 0:
                raise ValueError(f"Per-host batch {b} must be divisible by local_device_count {ldc}")

    # ──────────────────────────────────────────────────────────────────────────
    def __iter__(self):
        seen = 0

        # Apply skip if we're resuming from a checkpoint
        if self._skip_batches > 0:
            logging.info(
                f"Host {self._proc_idx}: Skipping {self._skip_batches} batches in this host's shard to resume from checkpoint..."
            )
            # Get the underlying dataset and apply skip
            underlying_ds = self._dataset._dataset
            if hasattr(underlying_ds, "dataset"):
                # For OXEDatasets and similar wrappers
                skipped_ds = underlying_ds.dataset.skip(self._skip_batches)
                underlying_ds.dataset = skipped_ds
            else:
                # For direct TF datasets
                self._dataset._dataset = underlying_ds.skip(self._skip_batches)

            self._skip_batches = 0  # Reset after applying skip
            logging.info(f"Host {self._proc_idx}: Skip complete, resuming training...")

        data_iter = iter(self._dataset)
        while True:
            if self._num_batches is not None and seen >= self._num_batches:
                return

            # Pull next preprocessed batch (may block on upstream I/O/TF)
            try:
                batch = next(data_iter)
            except StopIteration:
                # For validation, stop on StopIteration (don't restart iterator)
                # For training, restart the iterator to loop forever
                if self._split == "val":
                    return  # Stop iteration for validation
                data_iter = iter(self._dataset)
                continue

            self._assert_divisible(batch)
            batch = self._to_device(batch)
            seen += 1
            self._seen_batches += 1  # Track total batches seen for checkpointing
            yield CoTObservation.from_dict(batch), batch["actions"]

    def data_config(self) -> _config.DataConfig:
        return self._data_cfg

    @property
    def dataset(self) -> up.Dataset:
        return self._dataset._dataset

    @property
    def tokenizer(self) -> PaligemmaTokenizer:
        for t in self._dataset._transform.transforms:
            if hasattr(t, "tokenizer"):
                return t.tokenizer
        return None  # type: ignore

    def get_norm_stats_for_checkpoint(self) -> tuple[dict | None, str]:
        """Get normalization statistics to save with checkpoint.

        Returns:
            tuple: (norm_stats dict, description string)
            - For OXE with global normalization: (global_statistics, "global")
            - For OXE without global or DROID: (dataset_statistics, "per-dataset")
            - For unknown/unsupported: (None, "none")
        """
        underlying_dataset = self._dataset._dataset

        # For OXE datasets, prefer global statistics if available
        if hasattr(underlying_dataset, "global_statistics"):
            if underlying_dataset.global_statistics is not None:
                return underlying_dataset.global_statistics, "global"

        # Fall back to per-dataset statistics
        if hasattr(underlying_dataset, "dataset_statistics"):
            stats = underlying_dataset.dataset_statistics
            if stats is not None:
                return stats, "per-dataset"

        return None, "none"

    def save_dataloader_state(self, checkpoint_dir: str) -> str:
        """Save the dataloader state using batch counter for skip-based resumption.

        This uses a lightweight approach that saves only the batch counter,
        allowing resumption via dataset.skip(n).

        Args:
            checkpoint_dir: Directory to save the checkpoint file.
                           Supports both local paths and GCS paths (gs://...).

        Returns:
            The path to the saved checkpoint file.

        Note:
            - Saves only the batch counter (~8 bytes) for fast checkpointing
            - On resume, uses dataset.skip(n) to fast-forward to the checkpoint position
            - Works with all dataset types and operations
            - No persistent_iterator requirement
            - In multi-host setup, only host 0 saves to avoid race conditions

        Example:
            >>> loader.save_dataloader_state("./checkpoints/dataloader")
            './checkpoints/dataloader/dataloader_state.json'
            >>> loader.save_dataloader_state("gs://my-bucket/checkpoints/dataloader")
            'gs://my-bucket/checkpoints/dataloader/dataloader_state.json'
        """
        import json

        # # Only host 0 should save to avoid race conditions in multi-host setups
        # if self._proc_idx != 0:
        #     logging.info(f"Host {self._proc_idx}: Skipping dataloader state save (only host 0 saves)")
        #     return tf.io.gfile.join(checkpoint_dir, "dataloader_state.json")

        # Use tf.io.gfile for GCS compatibility
        if not tf.io.gfile.exists(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)

        # Save batch counter to JSON
        # Note: All hosts should have the same _seen_batches value due to synchronous training
        checkpoint_data = {
            "batches_seen": int(self._seen_batches),
            "version": "1.0",
        }

        checkpoint_path = tf.io.gfile.join(checkpoint_dir, "dataloader_state.json")
        logging.info(f"Host {self._proc_idx}: Saving dataloader state to {checkpoint_path}...")

        with tf.io.gfile.GFile(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logging.info(f"Host {self._proc_idx}: Saved dataloader state (batch {self._seen_batches})")
        return checkpoint_path

    def load_dataloader_state(self, checkpoint_dir: str) -> int:
        """Load the dataloader state from a checkpoint and prepare to skip batches.

        This method loads the batch counter and sets up the dataloader to skip
        the appropriate number of batches on the next call to __iter__.

        Args:
            checkpoint_dir: Directory containing the checkpoint file.
                           Supports both local paths and GCS paths (gs://...).

        Returns:
            The number of batches that were seen when the checkpoint was saved.

        Raises:
            ValueError: If no checkpoint file is found in the specified directory.

        Note:
            - Loads only the batch counter for lightweight checkpointing
            - The skip operation is deferred until __iter__ is called
            - Works with all dataset types and does not require persistent_iterator
            - After loading, the next iteration will automatically skip to the checkpoint position
            - All hosts load the same state and skip the same number of batches in their respective shards
            - Backward compatible: Falls back to process 0's checkpoint if current process checkpoint doesn't exist

        Example:
            >>> batches_seen = loader.load_dataloader_state("./checkpoints/dataloader")
            >>> print(f"Will resume from batch {batches_seen}")
            >>> batches_seen = loader.load_dataloader_state("gs://my-bucket/checkpoints/dataloader")
            >>> print(f"Will resume from batch {batches_seen}")
        """
        import json

        checkpoint_path = tf.io.gfile.join(checkpoint_dir, "dataloader_state.json")

        # If current process checkpoint doesn't exist, try process 0's checkpoint.
        if not tf.io.gfile.exists(checkpoint_path):
            # Extract parent directory and construct process 0 path
            # checkpoint_dir format: .../dataloader_process_{process_idx}
            parent_dir = tf.io.gfile.dirname(checkpoint_dir)
            process_0_dir = tf.io.gfile.join(parent_dir, "dataloader_process_0")
            process_0_checkpoint_path = tf.io.gfile.join(process_0_dir, "dataloader_state.json")

            if tf.io.gfile.exists(process_0_checkpoint_path):
                logging.warning(
                    f"Host {self._proc_idx}: Checkpoint not found at {checkpoint_path}, "
                    f"falling back to process 0 checkpoint"
                )
                checkpoint_path = process_0_checkpoint_path
            else:
                raise ValueError(f"No checkpoint file found at {checkpoint_path} or {process_0_checkpoint_path}")

        logging.info(f"Host {self._proc_idx}: Loading dataloader state from {checkpoint_path}...")

        with tf.io.gfile.GFile(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        self._seen_batches = checkpoint_data["batches_seen"]
        self._skip_batches = self._seen_batches  # Set skip counter for next iteration

        logging.info(f"Host {self._proc_idx}: Loaded dataloader state (batch {self._seen_batches})")
        logging.info(
            f"Host {self._proc_idx}: Will skip {self._skip_batches} batches in this host's shard on next iteration"
        )

        return self._seen_batches

    def get_batches_seen(self) -> int:
        """Get the number of batches seen so far.

        Returns:
            The count of batches processed.
        """
        return self._seen_batches

    def num_val_batches(self) -> int:
        """Get the configured number of batches per epoch.

        Returns:
            The number of batches if set, else None.
        """
        if hasattr(self.dataset, "num_val_batches_per_epoch"):
            return self.dataset.num_val_batches_per_epoch
        return 300  # Fallback value if not implemented
