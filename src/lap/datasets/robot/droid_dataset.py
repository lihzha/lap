"""DROID-specific dataset implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import tensorflow as tf

from lap.datasets.base_dataset import BaseRobotDataset
from lap.datasets.output_schema import TrajectoryOutputBuilder
from lap.datasets.registry import register_dataset
from lap.datasets.robot.droid_mixins import DroidLookupTableMixin
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import state_encoding_to_type

if TYPE_CHECKING:
    from lap.training.config import DataConfig


@register_dataset(name="droid", requires_hash_tables=True)
class DroidDataset(DroidLookupTableMixin, BaseRobotDataset):
    """DROID dataset with language action annotations.

    This dataset uses lookup tables for:
    - Episode ID mapping (file path -> episode ID)
    - Frame filtering (keep only high-quality frames)
    - Language instruction lookup
    """

    # DROID always requires wrist rotation
    NEEDS_WRIST_ROTATION: bool = True

    def __init__(
        self,
        *,  # Force keyword-only arguments
        data_dir: str,
        config: DataConfig,
        action_horizon: int = 16,
        action_dim: int = 32,
        state_dim: int = 10,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        seed: int = 0,
        split: str = "train",
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        hash_tables: dict = None,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        # Auto-configure parallel reads/calls based on available CPUs
        if num_parallel_calls == -1 or num_parallel_reads == -1:
            total_threads = len(os.sched_getaffinity(0))
            num_parallel_reads = int(total_threads * 0.3)
            num_parallel_calls = int(total_threads * 0.3)

        # Initialize DROID lookup tables using the mixin
        self._init_droid_tables(
            config=config,
            hash_tables=hash_tables,
            standalone=standalone,
            build_filter_table=True,
            build_instr_table=True,
        )

        super().__init__(
            dataset_name=config.droid_dataset_name,
            data_dir=data_dir,
            config=config,
            action_dim=action_dim,
            action_horizon=action_horizon,
            action_proprio_normalization_type=action_proprio_normalization_type,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
            standalone=standalone,
            shuffle=shuffle,
            batch_size=batch_size,
            max_samples=max_samples,
            enable_prediction_training=enable_prediction_training,
            pred_prob=pred_prob,
            primary_pred_prob=primary_pred_prob,
            state_dim=state_dim,
        )

    def get_traj_identifier(self):
        """Add trajectory ID from DROID episode lookup table."""

        def _get_traj_identifier(traj):
            episode_id = self.episode_id_from_traj(traj, self.ep_table)
            traj["trajectory_id"] = tf.fill([tf.shape(traj["action"])[0]], episode_id)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_restructure(self):
        """Restructure DROID trajectory to standardized format."""

        def restructure(traj):
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)

            actions = traj["action"]
            traj_len = tf.shape(actions)[0]
            episode_id = traj["trajectory_id"][0]

            # Randomly sample one of the three language instructions
            instruction = tf.random.shuffle(
                [
                    traj["language_instruction"],
                    traj["language_instruction_2"],
                    traj["language_instruction_3"],
                ]
            )[0]

            # Randomly sample one of the two exterior images
            random_val = tf.random.stateless_uniform(
                shape=[], seed=[self.seed, tf.strings.to_hash_bucket_fast(episode_id, 2147483647)]
            )
            exterior_img = tf.cond(
                random_val > 0.5,
                lambda: traj["observation"][self.spec.images_list[0]],
                lambda: traj["observation"][self.spec.images_list[1]],
            )

            # Build filter step IDs for frame-level filtering
            indices = tf.as_string(tf.range(traj_len))
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            # Build observation dict
            state_type_str = state_encoding_to_type(self.config.state_encoding)

            observation = {
                self.spec.primary_image_key: exterior_img,
                "state": tf.cast(traj["state"], tf.float32),
            }

            if self.use_wrist_image:
                observation[self.spec.wrist_image_key] = traj["observation"]["wrist_image_left"]
            else:
                observation[self.spec.wrist_image_key] = tf.repeat("", traj_len)

            # Include right wrist key for consistency with other datasets
            observation[self.spec.wrist_image_right_key] = tf.repeat("", traj_len)

            # Use output builder for consistent structure
            output = TrajectoryOutputBuilder.build_robot_trajectory(
                observation=observation,
                actions=actions,
                language_action=traj["language_action"],
                prompt=instruction,
                trajectory_id=traj["trajectory_id"],
                dataset_name=self.dataset_name,
                traj_len=traj_len,
                is_bimanual=False,  # DROID is single-arm
                state_type=state_type_str,
                has_wrist_image=True,
                needs_wrist_rotation=self.NEEDS_WRIST_ROTATION,
                is_navigation=False,
                extra_fields={
                    "traj_metadata": traj["traj_metadata"],
                    "passes_filter": passes_filter,
                },
            )

            # Rename language_instruction to prompt (DROID uses different prompt selection)
            output["prompt"] = output.pop("language_instruction")

            return output

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def apply_frame_filters(self):
        """Apply frame-level filter based on DROID keep_ranges."""

        def filter_from_dict(frame):
            return frame["passes_filter"]

        self.dataset = self.dataset.filter(filter_from_dict)

        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        self.dataset = self.dataset.map(remove_passes_filter)

        def _remove_raw_action(frame):
            frame.pop("language_action")
            return frame

        self.dataset = self.dataset.map(_remove_raw_action)

    def apply_traj_filters(self, action_key):
        """Apply trajectory-level filters for DROID."""

        # Filter out empty trajectories
        def _non_empty(traj):
            return tf.greater(tf.shape(traj[action_key])[0], 0)

        self.dataset = self.dataset.filter(_non_empty)

        # Filter out non-success trajectories
        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        # Filter out trajectories without language instructions
        def _has_instruction(traj):
            instr_bytes = self.instr_table.lookup(traj["trajectory_id"][0])
            return tf.logical_and(
                tf.not_equal(instr_bytes, tf.constant(b"", dtype=tf.string)),
                tf.greater(tf.strings.length(instr_bytes), 10),
            )

        self.dataset = self.dataset.filter(_path_ok)
        self.dataset = self.dataset.filter(_has_instruction)

    def apply_repack_transforms(self):
        """Repack DROID-specific fields."""
        super().apply_repack_transforms()

        def _pop_keys(traj):
            traj.pop("traj_metadata")
            return traj

        self.dataset = self.dataset.traj_map(_pop_keys, self.num_parallel_calls)
