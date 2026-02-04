from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import subprocess
import time
from typing import Protocol

from etils import epath
import jax
from openpi.shared import array_typing as at
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
import tensorflow as tf

from lap.datasets import data_loader as _data_loader
import lap.shared.normalize_adapter as _normalize_adapter
from lap.training import state as training_state


def _delete_gcs_prefix_with_gsutil(uri: str) -> None:
    """Delete a GCS URI prefix using gsutil recursively.

    Raises a CalledProcessError if deletion fails.
    """
    cmd = ["gsutil", "-m", "rm", "-r", uri]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logging.info("gsutil deletion output for %s:\n%s", uri, result.stdout)


def _make_async_options(async_enable: bool, async_timeout_secs: int | None) -> ocp.AsyncOptions | None:
    if not async_enable:
        return None
    kwargs = {}
    if async_timeout_secs is not None:
        kwargs["timeout_secs"] = async_timeout_secs
    return ocp.AsyncOptions(**kwargs)


def create_checkpoint_manager(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> ocp.CheckpointManager:
    checkpoint_dir = epath.Path(checkpoint_dir)
    async_options = _make_async_options(async_enable, async_timeout_secs)
    return ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=async_options,
        ),
    )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    async_timeout_secs: int | None = 7200,
    async_enable: bool = True,
) -> tuple[ocp.CheckpointManager, bool]:
    logging.info(f"Checkpoint_dir:{checkpoint_dir}")
    checkpoint_dir = epath.Path(checkpoint_dir)
    logging.info(f"Checkpoint_dir:{checkpoint_dir}")
    resuming = False
    is_gcs = str(checkpoint_dir).startswith("gs://")
    exists = tf.io.gfile.exists(str(checkpoint_dir)) if is_gcs else checkpoint_dir.exists()
    if exists:
        if overwrite:
            try:
                if is_gcs:
                    # Use gsutil to delete the GCS prefix recursively.
                    _delete_gcs_prefix_with_gsutil(str(checkpoint_dir))
                    # Recreate the prefix to ensure later writes succeed.
                    tf.io.gfile.makedirs(str(checkpoint_dir))
                else:
                    checkpoint_dir.rmtree()
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            except Exception as e:
                logging.warning(
                    "Failed to wipe checkpoint directory %s due to %s. Proceeding without wiping.",
                    checkpoint_dir,
                    e,
                )
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    if is_gcs:
        tf.io.gfile.makedirs(str(checkpoint_dir))
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = create_checkpoint_manager(
        checkpoint_dir,
        keep_period=keep_period,
        async_timeout_secs=async_timeout_secs,
        async_enable=async_enable,
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def _extract_directory(checkpoint_manager: ocp.CheckpointManager) -> str:
    directory_attr = getattr(checkpoint_manager, "directory", None)
    if directory_attr is None:
        directory_attr = getattr(checkpoint_manager, "_directory", None)
    return str(directory_attr) if directory_attr is not None else "<unknown>"


def _extract_keep_period(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    return getattr(options, "keep_period", None)


def _extract_async_timeout(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    async_opts = getattr(options, "async_options", None)
    if async_opts is None:
        return None
    return getattr(async_opts, "timeout_secs", None)


def _has_async_enabled(checkpoint_manager: ocp.CheckpointManager) -> bool:
    options = getattr(checkpoint_manager, "options", None)
    if options is None:
        options = getattr(checkpoint_manager, "_options", None)
    if options is None:
        return False
    return getattr(options, "async_options", None) is not None


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_state.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    max_retries: int = 0,
    retry_delay_secs: float = 0.0,
    retry_backoff: float = 1.0,
    fallback_to_sync: bool = False,
    async_timeout_secs: int | None = None,
    keep_period: int | None = None,
    preserve_checkpoint: bool = False,
) -> ocp.CheckpointManager:
    start_time = time.perf_counter()
    directory_str = _extract_directory(checkpoint_manager)
    logging.info("Checkpoint save start | step=%d | dir=%s | preserve=%s", step, directory_str, preserve_checkpoint)

    if keep_period is None:
        keep_period = _extract_keep_period(checkpoint_manager)
    if async_timeout_secs is None:
        async_timeout_secs = _extract_async_timeout(checkpoint_manager)

    # If preserve_checkpoint is True, save to a separate subdirectory to prevent
    # the main checkpoint manager's keep_period from deleting it
    if preserve_checkpoint:
        # Save to a separate "additional" subdirectory that's not managed by keep_period
        additional_dir = epath.Path(directory_str) / "additional"
        if str(additional_dir).startswith("gs://"):
            tf.io.gfile.makedirs(str(additional_dir))
        else:
            additional_dir.mkdir(parents=True, exist_ok=True)
        # Create a separate checkpoint manager for additional saves without keep_period
        manager_to_use = create_checkpoint_manager(
            str(additional_dir),
            keep_period=None,  # Disable keep_period to preserve these checkpoints
            async_timeout_secs=async_timeout_secs,
            async_enable=_has_async_enabled(checkpoint_manager),
        )
        logging.info("Saving preserved checkpoint at step %d to additional subdirectory: %s", step, additional_dir)
    else:
        manager_to_use = checkpoint_manager

    attempt = 0
    delay_secs = retry_delay_secs

    while True:
        attempt += 1
        try:

            def save_assets(directory: epath.Path):
                # Save the normalization stats from the dataset.
                # For OXE datasets with global normalization, this saves global statistics.
                # For DROID or OXE without global normalization, this saves per-dataset statistics.
                # Only process 0 saves normalization stats (same across all processes, no need to duplicate)
                if jax.process_index() == 0:
                    data_config = data_loader.data_config()
                    if hasattr(data_loader, "get_norm_stats_for_checkpoint"):
                        norm_stats, stats_type = data_loader.get_norm_stats_for_checkpoint()
                        if norm_stats is not None:
                            save_dir = str(directory / data_config.asset_id)
                            _normalize_adapter.save(save_dir, norm_stats)

                            # Log detailed information about saved statistics
                            stats_keys = list(norm_stats.keys())
                            logging.info(
                                f"Saved {stats_type} normalization statistics | keys={stats_keys} | location={save_dir}"
                            )

                            # Log detailed shape information for each key
                            for key, stat in norm_stats.items():
                                if hasattr(stat, "mean") and hasattr(stat.mean, "shape"):
                                    logging.info(
                                        f"  {key}: mean_shape={stat.mean.shape}, "
                                        f"num_transitions={getattr(stat, 'num_transitions', 'N/A')}, "
                                        f"num_trajectories={getattr(stat, 'num_trajectories', 'N/A')}"
                                    )
                        else:
                            logging.warning("No normalization statistics available to save with checkpoint")
                    else:
                        logging.info("Data loader does not support norm stats checkpointing (non-RLDS dataset)")

                # Save dataloader state (iterator position and batch counter)
                # This allows resuming training from the exact same data position
                # Saved to: {checkpoint_dir}/{step}/assets/dataloader_process_{process_index}/
                # - Same path structure as model checkpoint
                # - Same frequency as model checkpoint (save_interval)
                # - Automatically deleted with old checkpoints (keep_period)
                #
                # Multi-host behavior:
                # - Dataset is sharded per-host: dataset.shard(jax.process_count(), jax.process_index())
                # - Each host has a DIFFERENT iterator seeing different data shards
                # - Each host MUST save its own checkpoint to restore correctly
                # - All hosts save in parallel to their own subdirectories

                if hasattr(data_loader, "save_dataloader_state"):
                    try:
                        # directory is {checkpoint_dir}/{step}/assets/
                        # Each process saves to its own subdirectory
                        process_idx = jax.process_index()
                        dataloader_dir = str(directory / f"dataloader_process_{process_idx}")
                        save_path = data_loader.save_dataloader_state(dataloader_dir)
                        batches_seen = (
                            data_loader.get_batches_seen() if hasattr(data_loader, "get_batches_seen") else "unknown"
                        )
                        logging.info(
                            f"[Process {process_idx}] Saved dataloader state | batches_seen={batches_seen} | location={save_path}"
                        )
                    except Exception as e:
                        logging.warning(f"[Process {jax.process_index()}] Failed to save dataloader state: {e}")
                        logging.warning("Training will continue but dataloader state will not be checkpointed")
                elif jax.process_index() == 0:
                    logging.info("Data loader does not support state checkpointing (persistent_iterator not enabled)")

            # Split params that can be used for inference into a separate item.
            with at.disable_typechecking():
                train_state, params = _split_params(state)
            items = {
                "assets": save_assets,
                "train_state": train_state,
                "params": {"params": params},
            }
            manager_to_use.save(step, items)

            # Wait for async operations to complete before barrier
            # This ensures all async I/O (including dataloader state saves) has finished
            manager_to_use.wait_until_finished()

            # Multi-host barrier: Ensure all hosts wait for checkpoint save to complete
            # This is critical because all processes save their own dataloader state
            if jax.process_count() > 1:
                jax.experimental.multihost_utils.sync_global_devices("checkpoint_save_complete")

            duration = time.perf_counter() - start_time
            logging.info(
                "Checkpoint save complete | step=%d | dir=%s | duration=%.2fs",
                step,
                _extract_directory(manager_to_use),
                duration,
            )
            # Return the original checkpoint manager (not the temp one if preserve_checkpoint was used)
            return checkpoint_manager
        except KeyboardInterrupt:
            raise
        except Exception as err:
            duration = time.perf_counter() - start_time
            logging.warning(
                "Checkpoint save failed | step=%d | dir=%s | duration=%.2fs | attempt=%d | error=%s",
                step,
                _extract_directory(manager_to_use),
                duration,
                attempt,
                err,
            )
            if fallback_to_sync and manager_to_use is checkpoint_manager and _has_async_enabled(manager_to_use):
                logging.info("Retrying checkpoint save with synchronous manager (async disabled).")
                manager_to_use = create_checkpoint_manager(
                    _extract_directory(checkpoint_manager),
                    keep_period=keep_period,
                    async_timeout_secs=async_timeout_secs,
                    async_enable=False,
                )
                continue

            if attempt > max_retries:
                logging.error(
                    "Checkpoint save exhausted retries | step=%d | attempts=%d",
                    step,
                    attempt,
                )
                raise

            if delay_secs > 0:
                logging.info("Sleeping %.1fs before retrying checkpoint save", delay_secs)
                time.sleep(delay_secs)
                if retry_backoff > 1.0:
                    delay_secs *= retry_backoff


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_state.TrainState,
    data_loader: _data_loader.DataLoader | None,
    step: int | None = None,
    train_state_sharding: training_state.TrainState | None = None,
) -> training_state.TrainState:
    """Restore training state and dataloader state from checkpoint.

    Args:
        checkpoint_manager: The checkpoint manager
        state: Training state template to restore into
        data_loader: Data loader to restore iterator state
        step: Specific checkpoint step to restore (None = latest)
        train_state_sharding: Optional sharding tree to restore with explicit placement (e.g. evaluation)

    Returns:
        Restored training state
    """
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        if train_state_sharding is not None:
            train_state_sharding_without_params, params_sharding = _split_params(train_state_sharding)
            restore_args = ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    item=train_state,
                    transforms={},
                    restore_args=ocp.checkpoint_utils.construct_restore_args(
                        train_state, sharding_tree=train_state_sharding_without_params
                    ),
                ),
                params=ocp.args.PyTreeRestore(
                    item={"params": params},
                    transforms={},
                    restore_args=ocp.checkpoint_utils.construct_restore_args(
                        {"params": params}, sharding_tree={"params": params_sharding}
                    ),
                ),
            )
            restored = checkpoint_manager.restore(step, args=restore_args)
        else:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )

    # Restore dataloader state if available
    # Multi-host: Each host restores from its own checkpoint (saved per-process)
    # This ensures each host resumes its correct data shard position

    if data_loader is not None and hasattr(data_loader, "load_dataloader_state"):
        try:
            # Determine which step to restore from
            restore_step = step
            if restore_step is None:
                restore_step = checkpoint_manager.latest_step()

            # Construct dataloader checkpoint directory (same structure as save)
            # Each process has its own checkpoint: {checkpoint_dir}/{step}/assets/dataloader_process_{process_index}/
            checkpoint_dir = _extract_directory(checkpoint_manager)
            process_idx = jax.process_index()
            dataloader_dir = str(
                epath.Path(checkpoint_dir) / str(restore_step) / "assets" / f"dataloader_process_{process_idx}"
            )

            # Check if dataloader checkpoint exists for this process
            if tf.io.gfile.exists(dataloader_dir):
                # Each host restores from its own checkpoint
                batches_seen = data_loader.load_dataloader_state(dataloader_dir)
                logging.info(
                    f"[Process {process_idx}] Restored dataloader state | batches_seen={batches_seen} | step={restore_step} | location={dataloader_dir}"
                )
            else:
                logging.warning(
                    f"[Process {process_idx}] Dataloader checkpoint not found at {dataloader_dir}. "
                    "Dataloader will start from beginning of dataset shard."
                )

            # Multi-host barrier: Ensure all hosts have completed restore before continuing
            # This barrier must be OUTSIDE the exists check to prevent deadlock when some processes
            # have checkpoints while others don't
            if jax.process_count() > 1:
                jax.experimental.multihost_utils.sync_global_devices("dataloader_restore_complete")
                if jax.process_index() == 0:
                    logging.info(f"All {jax.process_count()} hosts synchronized after dataloader restore")
        except Exception as e:
            logging.warning(f"[Process {jax.process_index()}] Failed to restore dataloader state: {e}")
            logging.warning("Training will continue but dataloader will start from beginning")
    elif data_loader is not None and jax.process_index() == 0:
        logging.info("Data loader does not support state restoration (persistent_iterator not enabled)")

    return _merge_params(restored["train_state"], restored["params"])


def restore_params(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_state.TrainState,
    step: int | None = None,
    train_state_sharding: training_state.TrainState | None = None,
) -> training_state.TrainState:
    """Restore training state and dataloader state from checkpoint.

    Args:
        checkpoint_manager: The checkpoint manager
        state: Training state template to restore into
        data_loader: Data loader to restore iterator state
        step: Specific checkpoint step to restore (None = latest)
        train_state_sharding: Optional sharding tree to restore with explicit placement (e.g. evaluation)

    Returns:
        Restored training state
    """
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        _, params = _split_params(state)

        _, params_sharding = _split_params(train_state_sharding)
        restore_args = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                transforms={},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    {"params": params}, sharding_tree={"params": params_sharding}
                ),
            ),
        )
        restored = checkpoint_manager.restore(step, args=restore_args)

    return restored["params"]


def load_norm_stats(assets_dir: epath.Path | str) -> dict[str, _normalize_adapter.ExtendedNormStats] | None:
    """Load normalization stats from an assets directory.

    Preference order:
    1) A subdirectory named `combined`, `droid`, or `libero` that contains `norm_stats.json`.
    2) Any subdirectory that contains `norm_stats.json` (first match).
    """

    assets_dir = epath.Path(assets_dir)

    # Discover subdirectories that actually contain norm stats.
    norm_dirs = [p for p in assets_dir.iterdir() if p.is_dir() and (p / "norm_stats.json").exists()]

    assert len(norm_dirs) == 1, (
        f"Expected exactly one norm stats directory in {assets_dir}, but found {len(norm_dirs)}: "
        f"{[p.name for p in norm_dirs]}"
        )
    candidate = norm_dirs[0]
    logging.info(f"Loaded norm stats from {candidate}")

    return _normalize_adapter.load(str(candidate))


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        # Run callback on all processes to support per-process state saving (e.g., dataloader checkpoints)
        # Process-specific logic is handled inside the callback itself
        args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(
    state: training_state.TrainState,
) -> tuple[training_state.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(
    train_state: training_state.TrainState, params: dict[str, at.Params]
) -> training_state.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
