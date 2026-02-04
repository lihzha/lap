"""Training/validation metric aggregation and logging helpers."""

import logging
import os
from typing import TYPE_CHECKING, Union

from flax.training import common_utils
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
import psutil
import wandb

from lap.training import array_utils
from lap.training import batch_visualization

if TYPE_CHECKING:
    from lap.models.model_adapter import CoTObservation
    from lap.models.tokenizer import PaligemmaTokenizer
    import lap.training.config as _config


matplotlib.use("Agg")


class HostBatchCache:
    def __init__(self):
        self.host_batch: tuple[CoTObservation, _model.Actions] | None = None
        self.local_batch_size: int = 0
        self.step: int | None = None

    def ensure(
        self,
        *,
        step: int,
        batch: tuple["CoTObservation", _model.Actions],
    ) -> tuple[tuple["CoTObservation", _model.Actions] | None, int]:
        if self.step != step:
            self.host_batch = jax.tree.map(array_utils.to_local_array, batch)
            obs_local = self.host_batch[0] if self.host_batch else None
            self.local_batch_size = batch_visualization.infer_local_batch_size(obs_local)
            self.step = step
        return self.host_batch, self.local_batch_size


class DatasetLogTracker:
    """Tracks per-dataset logging counts to keep sample logging balanced."""

    def __init__(self, tokenizer: "PaligemmaTokenizer"):
        self.tokenizer = tokenizer
        self._dataset_log_counts: dict[str, int] = {}

    def get_dataset_names_from_batch(self, batch: tuple["CoTObservation", _model.Actions]) -> list[str]:
        obs = batch[0]
        if not hasattr(obs, "tokenized_dataset_name"):
            return []

        tokenized_names = array_utils.to_local_array(obs.tokenized_dataset_name)
        if tokenized_names is None:
            return []

        dataset_names = []
        for i in range(tokenized_names.shape[0]):
            try:
                name = self.tokenizer.decode(tokenized_names[i]).strip()
                dataset_names.append(name)
            except Exception as e:
                logging.warning(f"Failed to decode dataset name for sample {i}: {e}")
                dataset_names.append("unknown")

        return dataset_names

    def select_indices_uniform(
        self,
        dataset_names: list[str],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[int]:
        if not dataset_names or num_to_select <= 0:
            return []

        dataset_to_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(dataset_names):
            dataset_to_indices.setdefault(name, []).append(idx)

        datasets_sorted = sorted(dataset_to_indices.keys(), key=lambda d: self._dataset_log_counts.get(d, 0))

        selected_indices = []
        round_idx = 0
        while len(selected_indices) < num_to_select:
            added_any = False
            for dataset_name in datasets_sorted:
                if len(selected_indices) >= num_to_select:
                    break

                indices_for_dataset = dataset_to_indices[dataset_name]
                if round_idx == 0:
                    rng.shuffle(indices_for_dataset)

                if round_idx < len(indices_for_dataset):
                    selected_indices.append(indices_for_dataset[round_idx])
                    added_any = True

            if not added_any:
                break
            round_idx += 1

        return selected_indices

    def update_counts(self, dataset_names: list[str], selected_indices: list[int]) -> None:
        for idx in selected_indices:
            if idx < len(dataset_names):
                dataset_name = dataset_names[idx]
                self._dataset_log_counts[dataset_name] = self._dataset_log_counts.get(dataset_name, 0) + 1

    def get_stats(self) -> dict[str, int]:
        return dict(self._dataset_log_counts)


def log_mem(msg: str):
    """Log host RAM usage in GB."""
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024**3)
    logging.info(f"{msg}: RAM: {ram_gb:.2f}GB")


def log_random_examples(
    step: int,
    host_batch: tuple["CoTObservation", _model.Actions] | None,
    tokenizer: "PaligemmaTokenizer",
    *,
    local_batch_size: int,
    num_random: int = 5,
    dataset_log_tracker: DatasetLogTracker | None = None,
    prefix: str = "train",
) -> None:
    if host_batch is None or local_batch_size <= 0:
        return
    count = min(num_random, local_batch_size)
    if count <= 0:
        return

    process_idx = getattr(jax, "process_index", lambda: 0)()
    rng_seed = int(step + 997 * process_idx)
    rng_local = np.random.default_rng(rng_seed)

    if dataset_log_tracker is not None:
        dataset_names = dataset_log_tracker.get_dataset_names_from_batch(host_batch)
        if dataset_names:
            rand_idx = dataset_log_tracker.select_indices_uniform(dataset_names, count, rng_local)
            dataset_log_tracker.update_counts(dataset_names, rand_idx)
        else:
            rand_idx = rng_local.choice(local_batch_size, size=count, replace=False).tolist()
    else:
        rand_idx = rng_local.choice(local_batch_size, size=count, replace=False).tolist()

    random_visuals = batch_visualization.visualize_language_actions(
        host_batch,
        tokenizer,
        indices=rand_idx,
        max_examples=count,
    )
    if not random_visuals:
        return

    images_to_log = []
    for vis in random_visuals:
        caption_text = vis.get("prompt", "") or ""
        caption_text += vis.get("language_action", "") or ""
        caption_text += vis.get("dataset_name", "") or ""
        images_to_log.append(wandb.Image(vis["image"], caption=f"{caption_text}"))

    if images_to_log:
        wandb.log({f"{prefix}/random_examples": images_to_log}, step=step)


def process_and_log_metrics(
    step: int,
    infos: list[dict[str, at.Array]],
    batch: tuple[Union["CoTObservation", Observation], _model.Actions],
    config: "_config.TrainConfig",
    host_batch_cache: HostBatchCache | None = None,
    dataset_log_tracker: DatasetLogTracker | None = None,
    tok: Union["PaligemmaTokenizer", None] = None,
    prefix: str = "",
    verbose_mode: bool = False,
) -> dict[str, float]:
    """Process, aggregate and log training/validation metrics."""
    stacked_infos = common_utils.stack_forest(infos)
    reduce_overrides = {
        "grad_norm": jnp.mean,
        "loss": jnp.mean,
        "param_norm": jnp.mean,
    }
    reduced_info = {}

    for key, value in stacked_infos.items():
        if "per_sample_loss" in key:
            reduced_info[f"{prefix}max_{key}"] = jnp.max(value)
        elif "per_sample" in key:
            continue
        else:
            metric_key = key if key.startswith(prefix) else f"{prefix}{key}"
            reduced_info[metric_key] = reduce_overrides.get(key, jnp.mean)(value)

    reduced_info = jax.device_get(reduced_info)

    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
    mode = "val" if prefix else "train"
    logging.info(f"Step {step} ({mode}): {info_str}")

    if jax.process_index() == 0:
        wandb.log(reduced_info, step=step)

        if not prefix and host_batch_cache is not None and tok is not None and dataset_log_tracker is not None:
            host_batch_local, local_size = host_batch_cache.ensure(step=step, batch=batch)
            log_random_examples(
                step,
                host_batch_local,
                tok,
                local_batch_size=local_size,
                dataset_log_tracker=dataset_log_tracker,
                prefix=prefix[:-1] if prefix else "train",
            )

            if step % (config.log_interval * 10) == 0:
                log_stats = dataset_log_tracker.get_stats()
                if log_stats:
                    logging.info(f"Dataset logging counts: {log_stats}")
                    wandb_log_stats = {f"dataset_log_count/{name}": count for name, count in log_stats.items()}
                    wandb.log(wandb_log_stats, step=step)

    return reduced_info
