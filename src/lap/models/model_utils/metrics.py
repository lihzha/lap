import jax.numpy as jnp
from openpi.shared import array_typing as at

from lap.datasets.registry import VQA_DATASET_ID_TO_NAME


def compute_token_accuracy_metrics(
    predictions: at.Int[at.Array, "b s"],
    labels: at.Int[at.Array, "b s"],
    per_token_loss: at.Float[at.Array, "b s"],
    token_mask: at.Bool[at.Array, "b s"],
    critical_mask: at.Bool[at.Array, "b s"] | None = None,
    number_mask: at.Bool[at.Array, "b s"] | None = None,
    direction_mask: at.Bool[at.Array, "b s"] | None = None,
) -> dict[str, at.Array]:
    metrics = {}

    correct = (predictions == labels).astype(jnp.float32)
    masked_correct = correct * token_mask
    num_tokens = jnp.maximum(token_mask.sum(), 1.0)
    metrics["token_accuracy"] = masked_correct.sum() / num_tokens
    metrics["per_token_loss"] = per_token_loss
    metrics["labels"] = labels

    if critical_mask is not None:
        critical_correct = correct * critical_mask
        num_critical = jnp.maximum(critical_mask.sum(), 1.0)
        metrics["critical_token_accuracy"] = critical_correct.sum() / num_critical
        metrics["per_sample_critical_correct"] = critical_correct.sum(axis=-1)
        metrics["per_sample_critical_total"] = critical_mask.sum(axis=-1)

    if number_mask is not None:
        number_correct = correct * number_mask
        num_number = jnp.maximum(number_mask.sum(), 1.0)
        metrics["number_token_accuracy"] = number_correct.sum() / num_number
        metrics["per_sample_number_correct"] = number_correct.sum(axis=-1)
        metrics["per_sample_number_total"] = number_mask.sum(axis=-1)

    if direction_mask is not None:
        direction_correct = correct * direction_mask
        num_direction = jnp.maximum(direction_mask.sum(), 1.0)
        metrics["direction_token_accuracy"] = direction_correct.sum() / num_direction
        metrics["per_sample_direction_correct"] = direction_correct.sum(axis=-1)
        metrics["per_sample_direction_total"] = direction_mask.sum(axis=-1)

    return metrics


def compute_sample_specific_metrics(
    per_sample_loss: at.Float[at.Array, "b"],
    sample_mask: at.Bool[at.Array, "b"],
    prefix: str,
) -> dict[str, at.Array]:
    masked_loss = per_sample_loss * sample_mask
    num_samples = jnp.maximum(jnp.sum(sample_mask), 1.0)
    return {f"{prefix}loss": jnp.sum(masked_loss) / num_samples}


def compute_per_vqa_dataset_metrics(
    per_sample_loss: at.Float[at.Array, "b"],
    vqa_dataset_ids: at.Int[at.Array, "b"],
    vqa_mask: at.Bool[at.Array, "b"],
) -> dict[str, at.Array]:
    metrics = {}
    for dataset_id, dataset_name in VQA_DATASET_ID_TO_NAME.items():
        dataset_mask = jnp.logical_and(vqa_dataset_ids == dataset_id, vqa_mask)
        dataset_mask_f = dataset_mask.astype(jnp.float32)
        num_samples = jnp.sum(dataset_mask_f)
        masked_loss = per_sample_loss * dataset_mask_f
        dataset_loss = jnp.sum(masked_loss) / jnp.maximum(num_samples, 1.0)
        metrics[f"vqa_{dataset_name}_loss"] = dataset_loss
        metrics[f"vqa_{dataset_name}_num_samples"] = num_samples
    return metrics
