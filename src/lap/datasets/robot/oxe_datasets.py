"""OXE dataset implementations for RLDS datasets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import tensorflow as tf

from lap.datasets.base_dataset import BaseRobotDataset
from lap.datasets.output_schema import ObservationBuilder
from lap.datasets.output_schema import TrajectoryOutputBuilder
from lap.datasets.registry import needs_wrist_rotation
from lap.datasets.registry import register_dataset
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import state_encoding_to_type

if TYPE_CHECKING:
    from lap.training.config import DataConfig


class SingleOXEDataset(BaseRobotDataset):
    """Base class for OXE (Open X-Embodiment) datasets.

    This class provides the common functionality for OXE datasets:
    - Hash-based trajectory identification
    - Observation restructuring from raw RLDS format
    - Standard filtering (non-empty instructions, non-zero length)

    Subclasses can override specific behaviors by setting class attributes
    or overriding methods.
    """

    # Class attributes for dataset-specific behavior (override in subclasses)
    IS_NAVIGATION: bool = False
    FORCE_NO_WRIST_IMAGE: bool = False

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: DataConfig,
        action_horizon: int = 16,
        action_dim: int = 32,
        state_dim: int = 10,
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
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        # Centralized wrist rotation decision from registry configuration.
        self._needs_wrist_rotation = needs_wrist_rotation(dataset_name)

        super().__init__(
            dataset_name=dataset_name,
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

    def _build_observation(self, traj: dict, traj_len: tf.Tensor) -> dict:
        """Build observation dict from raw trajectory. Override for custom behavior."""
        return ObservationBuilder.build_from_image_keys(
            old_obs=traj["observation"],
            image_obs_keys=self.image_obs_keys,
            state_obs_keys=self.state_obs_keys,
            traj_len=traj_len,
            primary_image_key=self.spec.primary_image_key,
            wrist_image_key=self.spec.wrist_image_key,
            wrist_image_right_key=self.spec.wrist_image_right_key,
        )

    def _get_has_wrist_image(self) -> bool:
        """Get whether this dataset has wrist images. Override for custom behavior."""
        if self.FORCE_NO_WRIST_IMAGE:
            return False
        return self.has_wrist_image

    def _get_is_navigation(self) -> bool:
        """Get whether this is a navigation dataset. Override for custom behavior."""
        return self.IS_NAVIGATION

    def apply_restructure(self):
        """Restructure raw RLDS trajectory to standardized format."""

        def restructure(traj):
            traj_len = tf.shape(traj["action"])[0]
            new_obs = self._build_observation(traj, traj_len)
            state_type_str = state_encoding_to_type(self.state_encoding)

            return TrajectoryOutputBuilder.build_robot_trajectory(
                observation=new_obs,
                actions=traj["action"],
                language_action=traj["language_action"],
                prompt=traj["language_instruction"],
                trajectory_id=traj["trajectory_id"],
                dataset_name=self.dataset_name,
                traj_len=traj_len,
                is_bimanual=self.is_bimanual,
                state_type=state_type_str,
                has_wrist_image=self._get_has_wrist_image(),
                needs_wrist_rotation=self._needs_wrist_rotation,
                is_navigation=self._get_is_navigation(),
            )

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def get_traj_identifier(self):
        """Add trajectory ID using hash of dataset name and action sequence."""

        def _get_traj_identifier(traj):
            # Apply standardization function if provided
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)

            traj_len = tf.shape(traj["action"])[0]
            max_steps = 128

            # Use first/last 64 steps for long trajectories
            action_for_hash = tf.cond(
                max_steps >= traj_len,
                lambda: traj["action"],
                lambda: tf.concat([traj["action"][:64], traj["action"][-64:]], axis=0),
            )

            serialized_action = tf.io.serialize_tensor(action_for_hash)
            name_tensor = tf.constant(self.dataset_name, dtype=tf.string)
            to_hash = tf.strings.join([name_tensor, tf.constant("::", dtype=tf.string), serialized_action])
            hashed = tf.strings.to_hash_bucket_strong(to_hash, 2147483647, key=[self.seed, 1337])
            traj_uid = tf.strings.join([name_tensor, tf.constant("-", dtype=tf.string), tf.strings.as_string(hashed)])
            traj["trajectory_id"] = tf.repeat(traj_uid, traj_len)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_traj_filters(self, action_key):
        """Apply trajectory-level filters (non-empty instructions, non-zero length)."""

        def is_nonzero_length(traj):
            return tf.shape(traj[action_key])[0] > 0

        def has_any_instruction(traj):
            instr = traj["language_instruction"]
            instr = tf.reshape(instr, [-1])
            instr = tf.strings.strip(instr)
            return tf.reduce_any(tf.strings.length(instr) > 0)

        self.dataset = self.dataset.filter(has_any_instruction)
        self.dataset = self.dataset.filter(is_nonzero_length)

    def apply_frame_filters(self):
        """Apply frame-level filters (non-empty prompts)."""

        def _non_empty_prompt(frame: dict) -> tf.Tensor:
            p = tf.strings.strip(frame["prompt"])
            return tf.strings.length(p) > 0

        self.dataset = self.dataset.filter(_non_empty_prompt)

    def apply_repack_transforms(self):
        """Repack trajectory fields for final output format."""
        super().apply_repack_transforms()

        def _pop_and_rename_keys(traj):
            traj["prompt"] = traj["language_instruction"]
            traj.pop("language_instruction")
            traj.pop("language_action")
            return traj

        self.dataset = self.dataset.traj_map(_pop_and_rename_keys, self.num_parallel_calls)


@register_dataset(name="dobbe")
class DobbeDataset(SingleOXEDataset):
    """Dataset for dobbe with action range filtering.

    Filters out trajectories where any action exceeds [-5, 5] bounds.
    """

    # Dobbe action bounds
    ACTION_MIN: float = -5.0
    ACTION_MAX: float = 5.0

    def apply_traj_filters(self, action_key):
        """Apply trajectory filters including action range filter."""
        super().apply_traj_filters(action_key)

        min_allowed = self.ACTION_MIN
        max_allowed = self.ACTION_MAX

        def _action_within_bounds(traj):
            """Check if all actions are within bounds."""
            actions = traj[action_key]
            below_min = tf.reduce_any(tf.less(actions, min_allowed))
            above_max = tf.reduce_any(tf.greater(actions, max_allowed))
            return tf.logical_not(tf.logical_or(below_min, above_max))

        logging.info(f"Applying action range filter for dobbe: min={min_allowed}, max={max_allowed}")
        self.dataset = self.dataset.filter(_action_within_bounds)


@register_dataset(matcher=lambda n: "gnm_" in n, priority=10)
class NavigationDataset(SingleOXEDataset):
    """Dataset for navigation with 2D position state observations.

    Sets is_navigation=True and has_wrist_image=False.
    """

    IS_NAVIGATION: bool = True
    FORCE_NO_WRIST_IMAGE: bool = True


@register_dataset(matcher=lambda n: n.startswith("libero"), priority=10)
class LiberoDataset(SingleOXEDataset):
    """Dataset for LIBERO with EEF state observations.

    LIBERO has specific image key mappings and uses delta EEF actions.
    """

    def _build_observation(self, traj: dict, traj_len: tf.Tensor) -> dict:
        """Build observation dict with LIBERO-specific key mappings."""
        old_obs = traj["observation"]
        new_obs = {}

        # LIBERO uses: image (256x256x3), wrist_image (256x256x3)
        new_obs[self.spec.primary_image_key] = old_obs.get("image")
        new_obs[self.spec.wrist_image_key] = old_obs.get("wrist_image")
        # LIBERO doesn't have right wrist camera
        new_obs[self.spec.wrist_image_right_key] = tf.repeat("", traj_len)

        # State is already in the correct format (8D: [EEF pose (6D), gripper (2D)])
        new_obs["state"] = tf.cast(old_obs["state"], tf.float32)

        return new_obs

    def chunk_actions(self, traj, action_horizon, action_key="actions"):
        """LIBERO-specific action chunking with zero-padding."""
        from lap.datasets.utils.tfdata_pipeline import gather_with_padding

        traj_len = tf.shape(traj[action_key])[0]
        traj[action_key] = gather_with_padding(
            data=traj[action_key],
            sequence_length=traj_len,
            window_size=action_horizon,
        )
        return traj
