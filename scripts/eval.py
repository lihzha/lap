"""Evaluate token accuracy and/or rollout performance on a checkpoint using validation data."""

import dataclasses
import logging
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
from openpi.training import optimizer as _optimizer
from rail_tpu_utils import prevent_cross_region
import tqdm_loggable.auto as tqdm
import wandb

import lap.datasets.data_loader as _data_loader
from lap.models.model_adapter import CoTObservation
import lap.training.checkpoints as _checkpoints
import lap.training.config as _config
import lap.training.mh_sharding as sharding
import lap.training.state as training_state


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    # Only initialize wandb in the main process
    if jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    eval_tag = "eval"
    wandb.init(
        name=f"{config.exp_name}_{eval_tag}",
        config=dataclasses.asdict(config),
        project=config.project_name,
        tags=["evaluation"],
    )


def init_tpu(config: _config.TrainConfig):
    def _is_tpu_runtime() -> bool:
        try:
            return any(d.platform == "tpu" for d in jax.devices())
        except Exception:
            return False

    if config.fsdp_devices > 8:
        jax.distributed.initialize()

    data_dir = save_dir = config.data.rlds_data_dir
    cache_dir = os.environ.get("OPENPI_DATA_HOME", None)
    if _is_tpu_runtime() and (str(data_dir).startswith("gs://") or str(save_dir).startswith("gs://")):
        prevent_cross_region(data_dir, save_dir)
        if cache_dir is not None:
            prevent_cross_region(cache_dir, save_dir)
    # Determine effective FSDP devices for single-process GPU/CPU runs.
    process_count = getattr(jax, "process_count", lambda: 1)()
    local_devices = getattr(jax, "local_device_count", lambda: 1)()
    global_devices = getattr(jax, "device_count", lambda: local_devices)()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, local_devices)
        effective_fsdp_devices = 1
        for d in range(target, 0, -1):
            if global_devices % d == 0:
                effective_fsdp_devices = d
                break
        if effective_fsdp_devices != config.fsdp_devices:
            logging.info(
                "Using fsdp_devices=%d for single-process run (available devices=%d)",
                effective_fsdp_devices,
                global_devices,
            )
    else:
        effective_fsdp_devices = config.fsdp_devices

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    return effective_fsdp_devices


class ValidationLossEvaluator:
    """Validation loss evaluator that exactly matches train.py's ValidationStepRunner."""

    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_state.TrainState,
        batch: tuple[CoTObservation | Observation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch

        # Call compute_loss to get per-sample metrics for dataset tracking
        # Note: We use the model in eval mode but request per-sample metrics by passing train=True
        # This is to enable dataset-level tracking during validation
        # Pass verbose_mode=True to enable detailed metrics for validation
        verbose_mode = self.config.model.verbose_mode
        val_loss, val_metrics = model.compute_loss(
            eval_rng, observation, actions, train=False, verbose_mode=verbose_mode
        )

        val_metrics["val_loss"] = val_loss

        return val_metrics


class ActionPredictionLossEvaluator:
    """Evaluator that computes L2 loss between sampled actions and ground truth labels."""

    def __init__(self, config: _config.TrainConfig):
        self.config = config

    @at.typecheck
    def __call__(
        self,
        rng: at.KeyArrayLike,
        state: training_state.TrainState,
        batch: tuple[CoTObservation | Observation, _model.Actions],
    ) -> dict[str, at.Array]:
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        eval_rng = jax.random.fold_in(rng, state.step)
        observation, ground_truth_actions = batch

        # Sample actions from the model
        sample_rng, _ = jax.random.split(eval_rng)
        predicted_actions = model.sample_actions(sample_rng, observation)

        # Compute L2 loss (MSE) between predicted and ground truth actions
        # Shape: (batch, action_horizon, action_dim)
        # Compute mean over action_horizon and action_dim, keeping batch dimension
        per_sample_loss = jnp.mean(jnp.square(predicted_actions - ground_truth_actions), axis=(-1, -2))

        # Compute mean loss across batch
        mean_loss = jnp.mean(per_sample_loss)

        return {
            "action_prediction_loss": mean_loss,
            "per_sample_action_prediction_loss": per_sample_loss,
        }


def main(config: _config.TrainConfig):
    init_logging()
    effective_fsdp_devices = init_tpu(config)

    rng = jax.random.key(config.seed)
    eval_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Human-readable mesh overview
    sharding.log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", sharding.format_sharding(data_sharding))
    logging.info("Replicated sharding spec: %s", sharding.format_sharding(replicated_sharding))

    init_wandb(config, enabled=config.wandb_enabled)

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    ema_decay, ema_params_enabled = config.get_ema_init()

    def init(rng: at.KeyArrayLike) -> training_state.TrainState:
        # Initialize the model
        model = config.model.create(rng)
        params = nnx.state(model)

        return training_state.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=ema_decay,
            ema_params=None if not ema_params_enabled else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    train_state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # Initialize checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,  # Never overwrite for evaluation
        resume=True,  # Always resume for evaluation
        async_enable=False,  # No async for evaluation
    )

    # Log available checkpoints and determine which to load
    available_steps = sorted(list(checkpoint_manager.all_steps()))
    logging.info(f"Available checkpoints: {available_steps}")

    # Determine which checkpoints to evaluate
    if config.eval_all_checkpoints:
        checkpoint_steps_to_eval = available_steps
        logging.info(f"Evaluating all {len(checkpoint_steps_to_eval)} checkpoints: {checkpoint_steps_to_eval}")
    elif config.eval_checkpoint_steps is not None:
        checkpoint_steps_to_eval = config.eval_checkpoint_steps
        # Validate that all requested steps exist
        missing_steps = [step for step in checkpoint_steps_to_eval if step not in available_steps]
        if missing_steps:
            raise ValueError(
                f"Requested checkpoint steps {missing_steps} not found. Available steps: {available_steps}"
            )
        logging.info(f"Evaluating specified checkpoint steps: {checkpoint_steps_to_eval}")
    elif config.eval_checkpoint_step is not None:
        checkpoint_steps_to_eval = [config.eval_checkpoint_step]
        if checkpoint_steps_to_eval[0] not in available_steps:
            raise ValueError(
                f"Requested checkpoint step {checkpoint_steps_to_eval[0]} not found. Available steps: {available_steps}"
            )
        logging.info(f"Evaluating single checkpoint step: {checkpoint_steps_to_eval[0]}")
    else:
        latest_step = checkpoint_manager.latest_step()
        logging.info(f"No checkpoint step specified, using latest: {latest_step}")
        checkpoint_steps_to_eval = [latest_step]

    # Filter checkpoints based on eval_start_from_step (useful for resuming interrupted evaluations)
    if config.eval_start_from_step is not None:
        original_count = len(checkpoint_steps_to_eval)
        checkpoint_steps_to_eval = [step for step in checkpoint_steps_to_eval if step >= config.eval_start_from_step]
        skipped_count = original_count - len(checkpoint_steps_to_eval)
        if skipped_count > 0:
            logging.info(
                f"Skipping {skipped_count} checkpoints before step {config.eval_start_from_step}. "
                f"Remaining: {checkpoint_steps_to_eval}"
            )
        if not checkpoint_steps_to_eval:
            raise ValueError(
                f"No checkpoints found >= eval_start_from_step={config.eval_start_from_step}. "
                f"Available steps: {available_steps}"
            )

    # Determine number of evaluation batches (None means iterate until StopIteration)
    num_eval_batches = config.num_eval_batches

    # Define eval modes and dataset configs to iterate over
    eval_modes = ["val_loss", "action_prediction_loss"]

    # Create dataset configs: (config_name, modified_config)
    dataset_configs = [
        ("original", config),  # Use original config's data_mix and val_fraction
    ]

    # Add eval_mix config if data has data_mix attribute
    if hasattr(config.data, "data_mix") and config.data.data_mix is not None:
        # Create modified config with eval_mix and val_fraction=1.0
        eval_demo_config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                data_mix="eval_demo_dataset",
                val_fraction=1.0,
            ),
        )
        dataset_configs.append(("eval_demo_dataset", eval_demo_config))
        logging.info("Added eval_demo_dataset dataset configuration (data_mix='eval_demo_dataset', val_fraction=1.0)")

        # eval_rollout_config = dataclasses.replace(
        #     config,
        #     data=dataclasses.replace(
        #         config.data,
        #         data_mix="eval_rollout_dataset",
        #         val_fraction=1.0,
        #     ),
        # )
        # dataset_configs.append(("eval_rollout_dataset", eval_rollout_config))
        # logging.info("Added eval_rollout_dataset dataset configuration (data_mix='eval_rollout_dataset', val_fraction=1.0)")

    # Create data loaders once upfront (keyed by dataset_config_name)
    # This avoids recreating data loaders for each checkpoint, which is expensive
    logging.info("Creating data loaders for all dataset configurations...")
    data_loaders = {}
    for dataset_config_name, dataset_config in dataset_configs:
        logging.info(f"Creating data loader for dataset: {dataset_config_name}")
        data_loaders[dataset_config_name] = _data_loader.create_data_loader(
            dataset_config,
            sharding=data_sharding,
            shuffle=False,
            split=dataset_config.eval_split,
            seed=dataset_config.seed,
            max_samples=None,  # Don't limit samples for validation - iterate until StopIteration
        )
    logging.info(f"Created {len(data_loaders)} data loaders")

    # Evaluate each checkpoint sequentially
    all_results = {}
    for checkpoint_step in checkpoint_steps_to_eval:
        logging.info("=" * 80)
        logging.info(f"Evaluating checkpoint at step {checkpoint_step}")
        logging.info("=" * 80)

        # Restore checkpoint using the same helper as training (supports explicit sharding)
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state_shape,
            data_loader=None,
            step=checkpoint_step,
            train_state_sharding=train_state_sharding,
        )

        # Use EMA params for evaluation if available (matches training checkpoint layout)
        # Only use EMA params if checkpoint step >= ema_start_step
        ema_start_step = getattr(config.ema_schedule_choice, "start_step", 0) or 0
        if train_state.ema_params is not None and config.eval_use_ema:
            if checkpoint_step < ema_start_step:
                logging.info(
                    f"Checkpoint step {checkpoint_step} < ema_start_step {ema_start_step}, using original params"
                )
            else:
                logging.info(f"Checkpoint step {checkpoint_step} >= ema_start_step {ema_start_step}, using EMA params")
                train_state = dataclasses.replace(train_state, params=train_state.ema_params)

        logging.info(f"Loaded checkpoint at step {train_state.step}")
        sharding.log_param_sharding_actual(train_state.params)

        # Iterate over eval modes and dataset configs
        for eval_mode in eval_modes:
            for dataset_config_name, dataset_config in dataset_configs:
                logging.info("=" * 80)
                logging.info(f"Evaluating: mode={eval_mode}, dataset={dataset_config_name}")
                logging.info("=" * 80)

                # Get the pre-created data loader and create a fresh iterator for this checkpoint
                data_loader = data_loaders[dataset_config_name]
                data_iter = iter(data_loader)

                # Evaluate based on mode
                if eval_mode == "val_loss":
                    eval_results = evaluate_validation_loss(
                        dataset_config,
                        eval_rng,
                        train_state,
                        train_state_sharding,
                        data_iter,
                        mesh,
                        data_sharding,
                        replicated_sharding,
                        num_eval_batches,
                    )
                elif eval_mode == "action_prediction_loss":
                    eval_results = evaluate_action_prediction_loss(
                        dataset_config,
                        eval_rng,
                        train_state,
                        train_state_sharding,
                        data_iter,
                        mesh,
                        data_sharding,
                        replicated_sharding,
                        num_eval_batches,
                    )
                else:
                    raise ValueError(f"Unsupported eval_mode: {eval_mode}")

                # Store results with checkpoint step, eval mode, and dataset config
                prefix = f"step_{checkpoint_step}/{eval_mode}/{dataset_config_name}"
                step_results = {f"{prefix}/{k}": v for k, v in eval_results.items()}
                all_results.update(step_results)

                # Log results for this combination
                logging.info("=" * 80)
                logging.info(
                    f"EVALUATION RESULTS: step={checkpoint_step}, mode={eval_mode}, dataset={dataset_config_name}"
                )
                logging.info("=" * 80)
                for key, value in eval_results.items():
                    logging.info(f"{key:40s}: {value}")
                logging.info("=" * 80)

                # Log to wandb for this checkpoint step, eval mode, and dataset config
                if jax.process_index() == 0 and config.wandb_enabled:
                    # Add metadata to results for wandb
                    wandb_results = {f"{eval_mode}/{dataset_config_name}/{k}": v for k, v in eval_results.items()}
                    wandb.log(wandb_results, step=int(checkpoint_step))

    # Update wandb summary with all results
    if jax.process_index() == 0 and config.wandb_enabled:
        wandb.summary.update(all_results)

    return all_results


def evaluate_validation_loss(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_state.TrainState,
    train_state_sharding,
    data_iter,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int | None,
) -> dict[str, float]:
    """Evaluate validation loss, matching train.py's ValidationStepRunner exactly.

    Args:
        data_iter: Iterator over the data loader (caller should call iter() on the data loader).
        num_eval_batches: If None, iterate until StopIteration. Otherwise, limit to this many batches.
    """
    evaluator = ValidationLossEvaluator(config)
    pval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    val_infos = []

    # Use a progress bar that adapts to whether we have a fixed number of batches
    if num_eval_batches is not None:
        pbar = tqdm.tqdm(
            range(num_eval_batches),
            total=num_eval_batches,
            dynamic_ncols=True,
            desc="Validation loss evaluation",
            disable=(jax.process_index() != 0),
        )
        max_batches = num_eval_batches
    else:
        # For unknown number of batches, use a simple counter
        pbar = tqdm.tqdm(
            desc="Validation loss evaluation",
            dynamic_ncols=True,
            disable=(jax.process_index() != 0),
        )
        max_batches = None

    with sharding.set_mesh(mesh):
        batch_idx = 0
        while True:
            # Check if we've reached the batch limit
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of validation dataset at batch {batch_idx}")
                break

            val_info = pval_step(eval_rng, train_state, batch)
            val_infos.append(val_info)
            batch_idx += 1

            # Update progress bar
            if num_eval_batches is not None:
                pbar.update(1)
            else:
                pbar.update(1)
                pbar.total = batch_idx  # Update total as we go

            # Update progress bar with running loss
            if val_infos and batch_idx % 10 == 0:
                recent_losses = [
                    float(jax.device_get(info["val_loss"])) for info in val_infos[-min(10, len(val_infos)) :]
                ]
                pbar.set_postfix({"val_loss": f"{np.mean(recent_losses):.4f}", "batches": batch_idx})

        pbar.close()

    # Aggregate metrics across all batches
    results = {}
    if val_infos:
        # Get all metric keys from the first batch
        metric_keys = val_infos[0].keys()

        for key in metric_keys:
            # Skip non-scalar metrics
            values = []
            for info in val_infos:
                if key in info:
                    val = jax.device_get(info[key])
                    # Only aggregate scalar values
                    if np.isscalar(val) or (hasattr(val, "shape") and val.shape == ()):
                        values.append(float(val))
            if values:
                results[f"eval/val_loss/{key}"] = float(np.mean(values))

    results["eval/val_loss/num_batches"] = len(val_infos)

    return results


def evaluate_action_prediction_loss(
    config: _config.TrainConfig,
    eval_rng: at.KeyArrayLike,
    train_state: training_state.TrainState,
    train_state_sharding,
    data_iter,
    mesh,
    data_sharding,
    replicated_sharding,
    num_eval_batches: int | None,
) -> dict[str, float]:
    """Evaluate action prediction loss by sampling actions and comparing with labels.

    This computes the L2 loss between model-sampled actions and ground truth action labels.
    Only applies to action loss (not language loss).

    Args:
        data_iter: Iterator over the data loader (caller should call iter() on the data loader).
        num_eval_batches: If None, iterate until StopIteration. Otherwise, limit to this many batches.
    """
    evaluator = ActionPredictionLossEvaluator(config)
    pval_step = jax.jit(
        evaluator,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    val_infos = []

    # Use a progress bar that adapts to whether we have a fixed number of batches
    if num_eval_batches is not None:
        pbar = tqdm.tqdm(
            range(num_eval_batches),
            total=num_eval_batches,
            dynamic_ncols=True,
            desc="Action prediction loss evaluation",
            disable=(jax.process_index() != 0),
        )
        max_batches = num_eval_batches
    else:
        # For unknown number of batches, use a simple counter
        pbar = tqdm.tqdm(
            desc="Action prediction loss evaluation",
            dynamic_ncols=True,
            disable=(jax.process_index() != 0),
        )
        max_batches = None

    with sharding.set_mesh(mesh):
        batch_idx = 0
        while True:
            # Check if we've reached the batch limit
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                logging.info(f"Reached end of validation dataset at batch {batch_idx}")
                break

            val_info = pval_step(eval_rng, train_state, batch)
            val_infos.append(val_info)
            batch_idx += 1

            # Update progress bar
            if num_eval_batches is not None:
                pbar.update(1)
            else:
                pbar.update(1)
                pbar.total = batch_idx  # Update total as we go

            # Update progress bar with running loss
            if val_infos and batch_idx % 10 == 0:
                recent_losses = [
                    float(jax.device_get(info["action_prediction_loss"]))
                    for info in val_infos[-min(10, len(val_infos)) :]
                ]
                pbar.set_postfix({"action_pred_loss": f"{np.mean(recent_losses):.4f}", "batches": batch_idx})

        pbar.close()

    # Aggregate metrics across all batches
    results = {}
    if val_infos:
        # Get all metric keys from the first batch
        metric_keys = val_infos[0].keys()

        for key in metric_keys:
            # Skip non-scalar metrics
            values = []
            for info in val_infos:
                if key in info:
                    val = jax.device_get(info[key])
                    # Only aggregate scalar values
                    if np.isscalar(val) or (hasattr(val, "shape") and val.shape == ()):
                        values.append(float(val))
            if values:
                results[f"eval/action_prediction_loss/{key}"] = float(np.mean(values))

    results["eval/action_prediction_loss/num_batches"] = len(val_infos)

    return results


if __name__ == "__main__":
    config = _config.cli()
    main(config)
