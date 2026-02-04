from collections.abc import Iterable, Mapping, Sequence
import logging

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import jax
import numpy as np
from openpi.models import model as _model
import wandb

from lap.models.model_adapter import CoTObservation
from lap.models.tokenizer import PaligemmaTokenizer
from lap.training import array_utils


def infer_local_batch_size(obs_local: CoTObservation | None) -> int:
    if obs_local is None:
        return 0
    candidate_sizes: list[int] = []
    state_local = getattr(obs_local, "state", None)
    if state_local is not None:
        candidate_sizes.append(int(np.shape(state_local)[0]))
    prompt_local = getattr(obs_local, "tokenized_prompt", None)
    if prompt_local is not None:
        candidate_sizes.append(int(np.shape(prompt_local)[0]))
    image_values = list(getattr(obs_local, "images", {}).values())
    for img in image_values:
        if img is not None:
            candidate_sizes.append(int(np.shape(img)[0]))
            break
    return candidate_sizes[0] if candidate_sizes else 0


def _decode_reasoning_strings(obs: CoTObservation, tokenizer) -> tuple[list[str], list[str]]:
    """Extract and decode reasoning (language action) tokens per example."""
    tokens = array_utils.to_local_array(obs.tokenized_prompt)
    rmask = array_utils.to_local_array(obs.tokenized_prompt_mask)
    langact_mask = array_utils.to_local_array(obs.tokenized_langact_mask)
    texts: list[str] = []
    lang_acts: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        text = tokenizer.decode(sel.astype(np.int32))
        if langact_mask is None:
            lang_acts.append("")
        else:
            lang_act = tokens[i][langact_mask[i].astype(bool)]
            lang_act = tokenizer.decode(lang_act.astype(np.int32))
            lang_acts.append(lang_act)
        texts.append(text)
    return texts, lang_acts


def get_language_actions(batch, tok):
    texts, _ = _decode_reasoning_strings(batch[0], tok)
    return texts


def visualize_language_actions(
    batch: tuple[CoTObservation, _model.Actions],
    tok: PaligemmaTokenizer,
    *,
    indices: Sequence[int] | None = None,
    max_examples: int | None = 5,
    image_keys: Iterable[str] | None = None,
    resize_hw: tuple[int, int] | None = None,
) -> list[Mapping[str, object]]:
    """Return combined RGB images and decoded language actions for selected examples."""
    obs, _ = batch
    images = {key: array_utils.to_local_array(value) for key, value in obs.images.items() if value is not None}
    if not images:
        raise ValueError("No images found")

    order = list(image_keys) if image_keys is not None else sorted(images.keys())

    batch_sizes = [arr.shape[0] for arr in images.values() if arr is not None and arr.ndim >= 1]
    if not batch_sizes:
        raise ValueError("No images found")
    batch_size = min(batch_sizes)

    texts = get_language_actions(batch, tok)

    if indices is None:
        indices_list = list(range(batch_size))
    else:
        indices_list = [i for i in indices if 0 <= i < batch_size]

    if max_examples is not None:
        indices_list = indices_list[:max_examples]

    visuals: list[Mapping[str, object]] = []
    for idx in indices_list:
        per_cam: list[np.ndarray] = []
        for key in order:
            arr = images.get(key)
            if arr is None or idx >= arr.shape[0]:
                continue
            frame = np.asarray(arr[idx])
            if frame.ndim > 3:
                frame = frame[0]
            if np.issubdtype(frame.dtype, np.floating):
                frame = ((frame + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if resize_hw is not None and frame.shape[:2] != resize_hw:
                if HAS_CV2:
                    frame = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
                else:
                    h_old, w_old = frame.shape[:2]
                    h_new, w_new = resize_hw
                    row_idx = (np.arange(h_new) * h_old // h_new).astype(np.int32)
                    col_idx = (np.arange(w_new) * w_old // w_new).astype(np.int32)
                    frame = frame[row_idx[:, None], col_idx[None, :]]
            per_cam.append(frame)

        if not per_cam:
            continue

        if len(per_cam) == 1:
            combined = per_cam[0]
        else:
            try:
                combined = np.concatenate(per_cam, axis=1)
            except ValueError:
                max_h = max(img.shape[0] for img in per_cam)
                padded: list[np.ndarray] = []
                for img in per_cam:
                    if img.shape[0] == max_h:
                        padded.append(img)
                        continue
                    pad_total = max_h - img.shape[0]
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                    pad_spec = ((pad_top, pad_bottom), (0, 0), (0, 0))
                    padded_img = np.pad(img, pad_spec, mode="constant")
                    padded.append(padded_img)
                try:
                    combined = np.concatenate(padded, axis=1)
                except ValueError:
                    logging.warning("Failed to concatenate images for index %d due to incompatible shapes", idx)
                    combined = per_cam[0]

        text = texts[idx] if idx < len(texts) else ""
        visuals.append({"image": combined, "language_action": text, "index": idx})

    return visuals


def vis_batch(batch, tok=None, step=None):
    """Visualize a training batch for debugging purposes."""
    obs = batch[0]
    actions = batch[1]

    logging.info("=" * 80)
    logging.info("BATCH VISUALIZATION")
    logging.info("=" * 80)

    logging.info("\n--- IMAGES ---")
    wandb_images = {}
    for key, img in obs.images.items():
        logging.info(f"{key}: shape={img.shape}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")

        num_samples = img.shape[0]
        sample_images = []
        for t in range(min(num_samples, 4)):
            sample_img = img[t]
            sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            sample_img_uint8 = np.asarray(sample_img_uint8)
            sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_t{t}"))
            logging.info(
                f"  Prepared image [{key}] timestep {t} for wandb "
                f"(range: [{sample_img_uint8.min()}, {sample_img_uint8.max()}])"
            )

        if sample_images:
            wandb_images[f"batch_vis/{key}"] = sample_images

    logging.info("\n--- IMAGE MASKS ---")
    for key, mask in obs.image_masks.items():
        logging.info(f"{key}: shape={mask.shape}, dtype={mask.dtype}, true_count={mask.sum()}/{mask.size}")

    logging.info("\n--- STATE ---")
    state = obs.state
    logging.info(f"state: shape={state.shape}, dtype={state.dtype}")
    if len(state.shape) >= 2:
        for dim_idx in range(state.shape[-1]):
            dim_data = state[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    logging.info("\n--- TOKENIZED PROMPTS ---")
    tokenized_prompt = obs.tokenized_prompt
    tokenized_prompt_mask = obs.tokenized_prompt_mask

    logging.info(f"tokenized_prompt: shape={tokenized_prompt.shape}, dtype={tokenized_prompt.dtype}")
    logging.info(f"tokenized_prompt_mask: shape={tokenized_prompt_mask.shape}, dtype={tokenized_prompt_mask.dtype}")

    if tok is not None:
        sample_idx = 0
        if tokenized_prompt.shape[0] > 0:
            tokens_full = tokenized_prompt[sample_idx]
            decoded_full = tok.decode(tokens_full)
            logging.info(f"\n[Sample {sample_idx}] Full tokenized_prompt:")
            logging.info(f"  Decoded: {decoded_full}")

            tokens_masked = tokenized_prompt[sample_idx] * tokenized_prompt_mask[sample_idx]
            decoded_masked = tok.decode(tokens_masked)
            logging.info(f"\n[Sample {sample_idx}] tokenized_prompt * tokenized_prompt_mask:")
            logging.info(f"  Decoded: {decoded_masked}")
    else:
        logging.info("  (Tokenizer not provided - skipping decode)")

    logging.info("\n--- ACTIONS ---")
    logging.info(f"actions: shape={actions.shape}, dtype={actions.dtype}")
    if len(actions.shape) >= 2:
        for dim_idx in range(actions.shape[-1]):
            dim_data = actions[..., dim_idx]
            logging.info(
                f"  dim {dim_idx}: min={dim_data.min():.4f}, max={dim_data.max():.4f}, mean={dim_data.mean():.4f}"
            )

    logging.info("=" * 80)

    if wandb_images and jax.process_index() == 0 and step is not None:
        wandb.log(wandb_images, step=step)
        logging.info(f"Logged {len(wandb_images)} image groups to wandb")


def vis_augmented_images(
    augmented_images: dict[str, jax.Array] | None,
    step: int,
    prefix: str = "train",
    num_samples: int = 4,
) -> None:
    """Visualize augmented images from training step info."""
    if augmented_images is None or jax.process_index() != 0:
        return

    wandb_images = {}
    for key, img in augmented_images.items():
        img = array_utils.to_local_array(img)
        if img is None or img.ndim < 4:
            continue

        num_to_show = min(img.shape[0], num_samples)
        sample_images = []
        for t in range(num_to_show):
            sample_img = np.asarray(img[t])
            sample_img_uint8 = ((sample_img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            sample_images.append(wandb.Image(sample_img_uint8, caption=f"{key}_sample{t}"))

        if sample_images:
            wandb_images[f"{prefix}/augmented_{key}"] = sample_images

    if wandb_images:
        wandb.log(wandb_images, step=step)
        logging.info(f"Logged {len(wandb_images)} augmented image groups to wandb at step {step}")
