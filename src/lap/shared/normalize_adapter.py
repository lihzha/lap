import json
import logging
import os

import dlimp as dl
import jax
from jax.experimental import multihost_utils as mh
import numpy as np
import numpydantic
from openpi.shared import normalize as _normalize
import pydantic
import tensorflow as tf
from tqdm_loggable.auto import tqdm


@pydantic.dataclasses.dataclass
class ExtendedNormStats(_normalize.NormStats):
    num_transitions: int | None = None
    num_trajectories: int | None = None
    min: numpydantic.NDArray | None = None  # Global minimum across all samples
    max: numpydantic.NDArray | None = None  # Global maximum across all samples


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, ExtendedNormStats]


def serialize_json(norm_stats: dict[str, ExtendedNormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, ExtendedNormStats]:
    """Deserialize the running statistics from a JSON string."""
    # TODO: hard-coded for eef control mode
    raw_norm_stats = json.loads(data)
    raw_norm_stats["norm_stats"] = {
        k.replace("state_eef_pose", "state"): v for k, v in raw_norm_stats["norm_stats"].items()
    }

    return _NormStatsDict(**raw_norm_stats).norm_stats


def save(directory: str, norm_stats: dict[str, ExtendedNormStats]) -> None:
    """Save the normalization stats to a directory (supports gs:// or local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    tf.io.gfile.makedirs(os.path.dirname(str(path)))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(serialize_json(norm_stats))
    logging.info(f"Saved stats to: {path}")


def load(directory: str) -> dict[str, ExtendedNormStats]:
    """Load the normalization stats from a directory (supports gs:// and local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    with tf.io.gfile.GFile(path, "r") as f:
        return deserialize_json(f.read())


def check_dataset_statistics(save_dir: str | None = None) -> dict:
    """
    Checks if the dataset statistics are already computed and returns them if they are.
    """

    # Fallback local directory for when save_dir is not writable or not provided
    local_dir = os.path.expanduser(os.path.join("~", ".cache", "orca"))
    preferred_dir = save_dir if save_dir is not None else local_dir

    # Look for norm_stats.json in the preferred dir first, then in the local cache dir.
    if tf.io.gfile.exists(preferred_dir):
        try:
            stats = load(preferred_dir)
            return stats, preferred_dir, local_dir
        except Exception:
            pass

    if tf.io.gfile.exists(local_dir):
        try:
            stats = load(local_dir)
            return stats, local_dir, local_dir
        except Exception:
            pass

    return None, preferred_dir, local_dir


def get_dataset_statistics(
    dataset: dl.DLataset,
    save_dir: str | None = None,
    action_key: str = "action",
    state_key: str = "proprio",
) -> dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    _, output_dir, _ = check_dataset_statistics(save_dir)

    cardinality = dataset.cardinality().numpy()
    if cardinality in (tf.data.INFINITE_CARDINALITY, tf.data.UNKNOWN_CARDINALITY):
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality):
        actions_flat = traj[action_key].reshape(-1, traj[action_key].shape[-1])
        actions.append(actions_flat)
        proprios.append(traj["observation"][state_key])
        num_transitions += actions_flat.shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    mask = np.isfinite(actions).all(axis=1)
    actions = actions[mask]

    # Check if proprios has empty dimensions (state_encoding == NONE)
    has_state = proprios.shape[-1] > 0
    if has_state:
        mask = np.isfinite(proprios).all(axis=1)
        proprios = proprios[mask]

    # Promote to float64 for numerical stability in variance calculation
    # Float32 accumulation errors over millions of samples can cause variance to become
    # slightly negative (violating E[X^2] >= E[X]^2), which gets clamped to 0.
    # This produces std=0 even for dimensions with significant variation.
    # Example: 27M samples with range [0.27, 0.78] had std=0 in float32 but std=0.15 in float64
    actions = actions.astype(np.float64)
    if has_state:
        proprios = proprios.astype(np.float64)

    # ------------------------------------------------------------
    # Multi-host aggregation: compute exact global mean/std and counts
    # Using numerically stable variance calculation (Welford/Chan algorithm)
    # ------------------------------------------------------------
    def _gather_and_reduce(x: np.ndarray, op: str) -> np.ndarray:
        if getattr(jax, "process_count", lambda: 1)() == 1:
            return x
        xs = mh.process_allgather(np.asarray(x), tiled=False)  # shape: [P, ...]
        xs = np.asarray(xs)
        if op == "sum":
            return xs.sum(axis=0)
        if op == "min":
            return xs.min(axis=0)
        if op == "max":
            return xs.max(axis=0)
        raise ValueError(f"Unsupported op: {op}")

    # Compute numerically stable statistics using shifted values
    # Shift reduces magnitude of values, preventing catastrophic cancellation
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)

    if has_state:
        s_min = proprios.min(axis=0)
        s_max = proprios.max(axis=0)

    # Gather min/max across hosts to compute global shift
    a_min = _gather_and_reduce(a_min, "min")
    a_max = _gather_and_reduce(a_max, "max")

    if has_state:
        s_min = _gather_and_reduce(s_min, "min")
        s_max = _gather_and_reduce(s_max, "max")

    # Use midpoint as shift for numerical stability
    a_shift = (a_min + a_max) / 2.0

    if has_state:
        s_shift = (s_min + s_max) / 2.0

    # Compute shifted statistics (per-host)
    a_shifted = actions - a_shift
    a_sum = a_shifted.sum(axis=0)
    a_sumsq = np.square(a_shifted).sum(axis=0)
    a_n = np.array(actions.shape[0], dtype=np.int64)

    if has_state:
        s_shifted = proprios - s_shift
        s_sum = s_shifted.sum(axis=0)
        s_sumsq = np.square(s_shifted).sum(axis=0)
        s_n = np.array(proprios.shape[0], dtype=np.int64)

    traj_n = np.array(num_trajectories, dtype=np.int64)

    # All-gather + reduce shifted statistics
    a_sum = _gather_and_reduce(a_sum, "sum")
    a_sumsq = _gather_and_reduce(a_sumsq, "sum")
    a_n = int(_gather_and_reduce(a_n, "sum"))

    if has_state:
        s_sum = _gather_and_reduce(s_sum, "sum")
        s_sumsq = _gather_and_reduce(s_sumsq, "sum")
        s_n = int(_gather_and_reduce(s_n, "sum"))

    traj_n = int(_gather_and_reduce(traj_n, "sum"))

    # Compute global mean/std from shifted statistics
    # mean = shift + E[X']
    # var = E[X'²] - (E[X'])²  (now numerically stable due to shift)
    a_shifted_mean = a_sum / max(a_n, 1)
    a_mean = a_shift + a_shifted_mean
    a_var = a_sumsq / max(a_n, 1) - np.square(a_shifted_mean)

    # Check for negative variance (indicates numerical instability)
    if (a_var < 0).any():
        neg_dims = np.where(a_var < 0)[0]
        logging.warning(
            f"Action dims {neg_dims.tolist()} have negative variance {a_var[neg_dims].tolist()}, "
            f"clamping to 0. This indicates numerical instability - consider float64 promotion."
        )

    a_std = np.sqrt(np.maximum(a_var, 0.0))

    if has_state:
        s_shifted_mean = s_sum / max(s_n, 1)
        s_mean = s_shift + s_shifted_mean
        s_var = s_sumsq / max(s_n, 1) - np.square(s_shifted_mean)

        # Check for negative variance (indicates numerical instability)
        if (s_var < 0).any():
            neg_dims = np.where(s_var < 0)[0]
            logging.warning(
                f"State dims {neg_dims.tolist()} have negative variance {s_var[neg_dims].tolist()}, "
                f"clamping to 0. This indicates numerical instability - consider float64 promotion."
            )

        s_std = np.sqrt(np.maximum(s_var, 0.0))

    # ------------------------------------------------------------
    # Approximate global quantiles via distributed histograms
    # ------------------------------------------------------------
    def _distributed_quantiles(
        local_data: np.ndarray, g_min: np.ndarray, g_max: np.ndarray, q: float, num_bins: int = 4096
    ) -> np.ndarray:
        # Build identical bin edges per-dimension using global min/max
        dims = g_min.shape[0]
        if dims == 0:
            return np.array([], dtype=np.float32)
        edges = np.stack(
            [np.linspace(g_min[d] - 1e-12, g_max[d] + 1e-12, num_bins + 1) for d in range(dims)], axis=0
        )  # [D, B+1]
        local_hist = np.zeros((dims, num_bins), dtype=np.int64)
        for d in range(dims):
            # Guard against degenerate range
            if not np.isfinite(g_min[d]) or not np.isfinite(g_max[d]) or g_min[d] == g_max[d]:
                continue
            h, _ = np.histogram(local_data[:, d], bins=edges[d])
            local_hist[d] = h
        global_hist = _gather_and_reduce(local_hist, "sum")  # [D, B]
        # Compute q-quantile as left edge where cumsum crosses q * total
        q_vals = np.zeros((dims,), dtype=np.float64)
        for d in range(dims):
            counts = global_hist[d]
            total = counts.sum()
            if total == 0 or g_min[d] == g_max[d]:
                q_vals[d] = g_min[d]
                continue
            c = np.cumsum(counts)
            target = q * total
            idx = int(np.searchsorted(c, target, side="left"))
            if idx >= num_bins:
                idx = num_bins - 1
            q_vals[d] = edges[d, idx]
        return q_vals.astype(np.float32)

    a_q01 = _distributed_quantiles(actions, a_min, a_max, 0.01)
    a_q99 = _distributed_quantiles(actions, a_min, a_max, 0.99)

    if has_state:
        s_q01 = _distributed_quantiles(proprios, s_min, s_max, 0.01)
        s_q99 = _distributed_quantiles(proprios, s_min, s_max, 0.99)
        state_norm_stats = ExtendedNormStats(
            mean=np.asarray(s_mean, dtype=np.float32),
            std=np.asarray(s_std, dtype=np.float32),
            q01=np.asarray(s_q01, dtype=np.float32),
            q99=np.asarray(s_q99, dtype=np.float32),
            min=np.asarray(s_min, dtype=np.float32),
            max=np.asarray(s_max, dtype=np.float32),
            num_transitions=int(s_n),
            num_trajectories=int(traj_n),
        )
    else:
        # Create dummy state stats for empty state (state_encoding == NONE)
        state_norm_stats = ExtendedNormStats(
            mean=np.array([], dtype=np.float32),
            std=np.array([], dtype=np.float32),
            q01=np.array([], dtype=np.float32),
            q99=np.array([], dtype=np.float32),
            min=np.array([], dtype=np.float32),
            max=np.array([], dtype=np.float32),
            num_transitions=int(a_n),
            num_trajectories=int(traj_n),
        )

    # Cast back to float32 for storage efficiency (computed in float64 for precision)
    norm_stats = {
        "state": state_norm_stats,
        "actions": ExtendedNormStats(
            mean=np.asarray(a_mean, dtype=np.float32),
            std=np.asarray(a_std, dtype=np.float32),
            q01=np.asarray(a_q01, dtype=np.float32),
            q99=np.asarray(a_q99, dtype=np.float32),
            min=np.asarray(a_min, dtype=np.float32),
            max=np.asarray(a_max, dtype=np.float32),
            num_transitions=int(a_n),
            num_trajectories=int(traj_n),
        ),
    }

    if jax.process_index() == 0:
        logging.info("Dataset statistics computed:")
        logging.info(f"  Total transitions: {a_n}, trajectories: {traj_n}")
        logging.info(f"  Actions ({len(a_mean)} dims):")
        logging.info(f"    min: {a_min}")
        logging.info(f"    max: {a_max}")
        logging.info(f"    mean: {a_mean}")
        logging.info(f"    std: {a_std}")

        # Check for std=0 dimensions
        zero_action_dims = np.where(a_std == 0)[0]
        if len(zero_action_dims) > 0:
            logging.warning(f"    ⚠️  {len(zero_action_dims)} action dims have std=0: {zero_action_dims.tolist()}")

        if has_state and len(s_mean) > 0:
            logging.info(f"  State ({len(s_mean)} dims):")
            logging.info(f"    min: {s_min}")
            logging.info(f"    max: {s_max}")
            logging.info(f"    mean: {s_mean}")
            logging.info(f"    std: {s_std}")

            zero_state_dims = np.where(s_std == 0)[0]
            if len(zero_state_dims) > 0:
                logging.warning(f"    ⚠️  {len(zero_state_dims)} state dims have std=0: {zero_state_dims.tolist()}")

        logging.info(f"Writing stats to: {output_dir}")
        save(output_dir, norm_stats)

    return norm_stats
