"""Global statistics computation utilities.

This module provides utilities for computing weighted global statistics
across multiple datasets, extracted from dataset_mixer.py for better
modularity and testability.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class GlobalStatisticsBuilder:
    """Builder for computing weighted global statistics across datasets.

    This class handles the computation of global normalization statistics
    when mixing multiple datasets, accounting for:
    - Different dataset sizes (weighted by num_transitions)
    - Different state types (joint_pos, eef_pose, none)
    - Proper variance computation using parallel algorithm

    Usage:
        builder = GlobalStatisticsBuilder(action_dim=32, state_dim=10)
        global_stats = builder.compute_global_stats(
            all_dataset_statistics={"droid": stats1, "bridge": stats2},
            dataset_state_encodings={"droid": StateEncoding.POS_EULER, ...},
            vqa_dataset_names={"coco_captions"},
        )
    """

    def __init__(self, action_dim: int, state_dim: int):
        """Initialize the builder.

        Args:
            action_dim: Dimension of action space.
            state_dim: Dimension of state space.
        """
        self.action_dim = action_dim
        self.state_dim = state_dim

    def compute_global_stats(
        self,
        all_dataset_statistics: dict,
        dataset_state_encodings: dict,
        vqa_dataset_names: set[str],
        state_encoding_to_type_fn=None,
    ) -> dict:
        """Compute global normalization statistics across all datasets.

        Args:
            all_dataset_statistics: Dict mapping dataset_name -> statistics dict.
            dataset_state_encodings: Dict mapping dataset_name -> StateEncoding.
            vqa_dataset_names: Set of VQA dataset names to exclude from stats.
            state_encoding_to_type_fn: Function to convert StateEncoding to type string.

        Returns:
            Dict with global statistics for actions and per-state-type states.
        """
        from lap.datasets.utils.helpers import state_encoding_to_type

        if state_encoding_to_type_fn is None:
            state_encoding_to_type_fn = state_encoding_to_type

        # Group datasets by state type
        datasets_by_state_type = {"joint_pos": [], "eef_pose": [], "none": []}
        for dataset_name in all_dataset_statistics:
            if dataset_name in vqa_dataset_names:
                continue
            state_encoding = dataset_state_encodings.get(dataset_name)
            if state_encoding is not None:
                state_type = state_encoding_to_type_fn(state_encoding)
                datasets_by_state_type[state_type].append(dataset_name)

        # Compute action statistics
        action_stats = self._compute_action_stats(all_dataset_statistics, vqa_dataset_names)

        global_stats = {"actions": action_stats}

        # Compute state statistics per type
        for state_type, ds_names in datasets_by_state_type.items():
            if not ds_names or state_type == "none":
                continue

            state_stats = self._compute_state_stats(ds_names, all_dataset_statistics)
            if state_stats is not None:
                global_stats[f"state_{state_type}"] = state_stats

        return global_stats

    def _compute_action_stats(
        self,
        all_dataset_statistics: dict,
        vqa_dataset_names: set[str],
    ):
        """Compute weighted global action statistics."""
        from lap.shared.normalize_adapter import ExtendedNormStats

        total_n = 0
        weighted_sum = np.zeros(self.action_dim, dtype=np.float32)

        # First pass: compute weighted mean
        for dataset_name, stats in all_dataset_statistics.items():
            if dataset_name in vqa_dataset_names:
                continue
            n = stats["actions"].num_transitions
            total_n += n
            mean_padded = self._pad_to_dim(stats["actions"].mean, self.action_dim)
            weighted_sum += mean_padded * n

        if total_n == 0:
            return ExtendedNormStats(
                mean=np.zeros(self.action_dim, dtype=np.float32),
                std=np.ones(self.action_dim, dtype=np.float32),
                q01=np.zeros(self.action_dim, dtype=np.float32),
                q99=np.zeros(self.action_dim, dtype=np.float32),
                num_transitions=0,
                num_trajectories=0,
            )

        global_mean = weighted_sum / total_n

        # Second pass: compute weighted variance
        var_sum = np.zeros_like(global_mean)
        for dataset_name, stats in all_dataset_statistics.items():
            if dataset_name in vqa_dataset_names:
                continue
            n = stats["actions"].num_transitions
            local_mean = self._pad_to_dim(stats["actions"].mean, self.action_dim)
            local_std = self._pad_to_dim(stats["actions"].std, self.action_dim, pad_value=0.0)
            local_var = np.square(local_std)
            mean_diff_sq = np.square(local_mean - global_mean)
            var_sum += n * (local_var + mean_diff_sq)

        global_var = var_sum / total_n
        global_std = np.sqrt(global_var)

        # Compute quantiles and min/max
        q01_list, q99_list, min_list, max_list = [], [], [], []
        for dataset_name, stats in all_dataset_statistics.items():
            if dataset_name in vqa_dataset_names:
                continue
            q01_list.append(self._pad_to_dim(stats["actions"].q01, self.action_dim))
            q99_list.append(self._pad_to_dim(stats["actions"].q99, self.action_dim))
            min_list.append(self._pad_to_dim(stats["actions"].min, self.action_dim))
            max_list.append(self._pad_to_dim(stats["actions"].max, self.action_dim))

        global_q01 = np.min(q01_list, axis=0) if q01_list else np.zeros(self.action_dim)
        global_q99 = np.max(q99_list, axis=0) if q99_list else np.zeros(self.action_dim)
        global_min = np.min(min_list, axis=0) if min_list else np.zeros(self.action_dim)
        global_max = np.max(max_list, axis=0) if max_list else np.zeros(self.action_dim)

        return ExtendedNormStats(
            mean=global_mean,
            std=global_std,
            q01=global_q01,
            q99=global_q99,
            min=global_min,
            max=global_max,
            num_transitions=total_n,
            num_trajectories=sum(
                stats["actions"].num_trajectories
                for name, stats in all_dataset_statistics.items()
                if name not in vqa_dataset_names
            ),
        )

    def _compute_state_stats(
        self,
        ds_names: list[str],
        all_dataset_statistics: dict,
    ):
        """Compute weighted state statistics for a subset of datasets."""
        from lap.shared.normalize_adapter import ExtendedNormStats

        state_stats_subset = {name: all_dataset_statistics[name] for name in ds_names}
        total_n = sum(stats["state"].num_transitions for stats in state_stats_subset.values())

        if total_n == 0:
            return None

        # Compute weighted mean
        weighted_sum = np.zeros(self.state_dim, dtype=np.float32)
        for stats in state_stats_subset.values():
            n = stats["state"].num_transitions
            mean_padded = self._pad_to_dim(stats["state"].mean, self.state_dim)
            weighted_sum += mean_padded * n

        global_mean = weighted_sum / total_n

        # Compute weighted variance
        var_sum = np.zeros_like(global_mean)
        for stats in state_stats_subset.values():
            n = stats["state"].num_transitions
            local_mean = self._pad_to_dim(stats["state"].mean, self.state_dim)
            local_std = self._pad_to_dim(stats["state"].std, self.state_dim, pad_value=0.0)
            local_var = np.square(local_std)
            mean_diff_sq = np.square(local_mean - global_mean)
            var_sum += n * (local_var + mean_diff_sq)

        global_var = var_sum / total_n
        global_std = np.sqrt(global_var)

        # Compute quantiles and min/max
        q01_list = [self._pad_to_dim(stats["state"].q01, self.state_dim) for stats in state_stats_subset.values()]
        q99_list = [self._pad_to_dim(stats["state"].q99, self.state_dim) for stats in state_stats_subset.values()]
        min_list = [self._pad_to_dim(stats["state"].min, self.state_dim) for stats in state_stats_subset.values()]
        max_list = [self._pad_to_dim(stats["state"].max, self.state_dim) for stats in state_stats_subset.values()]

        return ExtendedNormStats(
            mean=global_mean,
            std=global_std,
            q01=np.min(q01_list, axis=0),
            q99=np.max(q99_list, axis=0),
            min=np.min(min_list, axis=0),
            max=np.max(max_list, axis=0),
            num_transitions=total_n,
            num_trajectories=sum(stats["state"].num_trajectories for stats in state_stats_subset.values()),
        )

    def _pad_to_dim(self, arr: np.ndarray, target_dim: int, pad_value: float = 0.0) -> np.ndarray:
        """Pad array to target dimension."""
        if len(arr) >= target_dim:
            return arr[:target_dim]
        return np.pad(arr, (0, target_dim - len(arr)), mode="constant", constant_values=pad_value)

    @staticmethod
    def log_global_stats(
        name: str,
        dim: int,
        global_min: np.ndarray,
        global_max: np.ndarray,
        q01: np.ndarray,
        q99: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        """Log global statistics per dimension."""
        logging.info("=" * 80)
        logging.info(f"Global {name} Statistics per Dimension:")
        logging.info("-" * 80)
        for dim_idx in range(dim):
            logging.info(
                f"{name} dim {dim_idx:2d}: "
                f"min={global_min[dim_idx]:9.4f}, "
                f"max={global_max[dim_idx]:9.4f}, "
                f"q01={q01[dim_idx]:9.4f}, "
                f"q99={q99[dim_idx]:9.4f}, "
                f"mean={mean[dim_idx]:9.4f}, "
                f"std={std[dim_idx]:8.4f}"
            )
        logging.info("=" * 80)
