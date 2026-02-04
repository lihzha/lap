"""
data_utils.py

Additional RLDS-specific data utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import dlimp as dl
import numpy as np
import tensorflow as tf

from lap.datasets.utils.configs import OXE_DATASET_CONFIGS
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import ActionEncoding
from lap.datasets.utils.transforms import OXE_STANDARDIZATION_TRANSFORMS


def tree_map(fn: Callable, tree: dict) -> dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: dict) -> dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    if tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


# === State / Action Processing Primitives ===


def normalize_action_and_proprio(
    traj: dict,
    norm_stats=None,
    normalization_type: NormalizationType | str = NormalizationType.NORMAL,
    action_key: str = "action",
    state_key: str = "proprio",
    metadata=None,  # Alias for callers that pass `metadata` instead of `norm_stats`
):
    """Normalizes the action and proprio fields of a trajectory using provided stats.

    Accepts either `norm_stats` or `metadata`. Supports
    stats structures with keys "actions" or "action", and values that may be
    dicts, dataclass-like objects (with attributes), lists, NumPy arrays or
    tensors. All stats are coerced to tf.Tensor before use.
    """

    # Support alias argument name
    if norm_stats is None and metadata is not None:
        norm_stats = metadata

    if norm_stats is None:
        return traj

    # Normalize enum input
    if isinstance(normalization_type, str):
        try:
            normalization_type = NormalizationType(normalization_type)
        except ValueError:
            raise ValueError(f"Unknown Normalization Type {normalization_type}")

    def _get_group(stats_root, group_name: str):
        # support both "actions" and "action"
        if isinstance(stats_root, dict):
            group = stats_root.get(group_name)
            if group is None and group_name.endswith("s"):
                group = stats_root.get(group_name[:-1])
            return group
        # if a non-dict is provided, there's nothing to select
        return None

    def _get_value(group_stats, key: str):
        if group_stats is None:
            return None
        if isinstance(group_stats, dict):
            value = group_stats.get(key)
        else:
            # dataclass-like objects
            value = getattr(group_stats, key, None)
        if value is None:
            return None
        # Coerce to tf.Tensor[float32]
        if isinstance(value, tf.Tensor):
            return tf.cast(value, tf.float32)
        return tf.convert_to_tensor(value, dtype=tf.float32)

    def normal(x, mean, std):
        return (x - mean) / (std + 1e-6)

    def bounds(x, _min, _max):
        return 2 * (x - _min) / (_max - _min + 1e-6) - 1

    actions_stats = _get_group(norm_stats, "actions")
    state_stats = _get_group(norm_stats, "state")

    if normalization_type == NormalizationType.NORMAL:
        a_mean = _get_value(actions_stats, "mean")
        a_std = _get_value(actions_stats, "std")
        s_mean = _get_value(state_stats, "mean")
        s_std = _get_value(state_stats, "std")

        if a_mean is not None and a_std is not None:
            traj[action_key] = normal(traj[action_key], a_mean, a_std)
        if s_mean is not None and s_std is not None and state_key in traj.get("observation", {}):
            traj["observation"][state_key] = normal(traj["observation"][state_key], s_mean, s_std)
    elif normalization_type in (NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99):
        low_key = "min" if normalization_type == NormalizationType.BOUNDS else "q01"
        high_key = "max" if normalization_type == NormalizationType.BOUNDS else "q99"

        action_low = _get_value(actions_stats, low_key)
        action_high = _get_value(actions_stats, high_key)
        state_low = _get_value(state_stats, low_key)
        state_high = _get_value(state_stats, high_key)

        if action_low is not None and action_high is not None:
            traj[action_key] = bounds(traj[action_key], action_low, action_high)
            zeros_mask = tf.equal(action_low, action_high)
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == action_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        if state_low is not None and state_high is not None and state_key in traj.get("observation", {}):
            traj["observation"][state_key] = bounds(traj["observation"][state_key], state_low, state_high)
            zeros_mask = tf.equal(state_low, state_high)
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == state_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

    return traj


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_names: list[str], dataset_weights: list[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_names)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_name, weight in zip(dataset_names, dataset_weights):
        pad = 80 - len(dataset_name)
        print(f"# {dataset_name}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def allocate_threads(n: int | None, weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation


def load_dataset_kwargs(
    dataset_name: str,
    rlds_data_dir: Path,
    load_camera_views: tuple[str] = ("primary", "wrist"),
) -> dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    # Support multiple action encodings: EEF_POS, EEF_R6, ABS_EEF_POS, JOINT_POS, JOINT_POS_BIMANUAL
    supported_encodings = [
        ActionEncoding.EEF_POS,
        ActionEncoding.EEF_R6,
        ActionEncoding.ABS_EEF_POS,
        ActionEncoding.JOINT_POS,
        ActionEncoding.JOINT_POS_BIMANUAL,
    ]
    if dataset_kwargs["action_encoding"] not in supported_encodings:
        raise ValueError(
            f"Cannot load `{dataset_name}`; action encoding {dataset_kwargs['action_encoding']} not supported! "
            f"Supported encodings: {supported_encodings}"
        )

    language_annotations = dataset_kwargs.get("language_annotations")
    if not language_annotations or language_annotations.lower() == "none":
        raise ValueError(f"Cannot load `{dataset_name}`; language annotations required!")

    robot_morphology = dataset_kwargs.get("robot_morphology", "")
    is_bimanual = robot_morphology.lower() == "bi-manual"

    # Add bimanual flag to dataset kwargs
    dataset_kwargs["is_bimanual"] = is_bimanual

    has_suboptimal = dataset_kwargs.get("has_suboptimal")
    if isinstance(has_suboptimal, str):
        has_suboptimal = has_suboptimal.lower() == "yes"
    if has_suboptimal:
        logging.warning(f"Cannot load `{dataset_name}`; suboptimal datasets are not supported!")

    # Filter
    dataset_kwargs["image_obs_keys"] = {k: dataset_kwargs["image_obs_keys"].get(k, None) for k in load_camera_views}

    # dataset_kwargs["image_obs_keys"] = {
    #     k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in camera_views_to_load
    # }
    for k, v in dataset_kwargs["image_obs_keys"].items():
        if k == "primary":
            assert v is not None, f"primary image is required for {dataset_name}"

    # Specify Standardization Transform
    # Use unified registry (superset), still supports all OXE datasets
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(rlds_data_dir), **dataset_kwargs}
