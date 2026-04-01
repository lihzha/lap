import logging
import os
import platform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import etils.epath as epath
import jax
import jax.experimental.multihost_utils as multihost_utils
import numpy as np
from rail_tpu_utils import prevent_cross_region

import lap.datasets.data_loader as _data_loader
from lap.models.tokenizer import PaligemmaTokenizer
import lap.training.config as _config
import lap.training.mh_sharding as sharding

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional dependency when running offline
    wandb = None  # type: ignore[assignment]


def _safe_device_get(arr):
    """Safely get array to host, handling multi-host sharded arrays."""
    if arr is None:
        return None
    try:
        # Try direct device_get first (works for single-host or local shards)
        return jax.device_get(arr)
    except RuntimeError as e:
        if "non-addressable" in str(e):
            # Array spans multiple hosts, need to gather
            try:
                gathered = multihost_utils.process_allgather(arr, tiled=True)
                return jax.device_get(gathered)
            except Exception as gather_error:
                logging.warning(f"Failed to gather array: {gather_error}")
                return None
        raise


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

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


def _format_sharding(shard) -> str:
    try:
        import jax
    except Exception:
        return "<no-jax>"
    if isinstance(shard, jax.sharding.NamedSharding):
        mesh = shard.mesh
        mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
        return f"NamedSharding(mesh=[{mesh_desc}], spec={shard.spec})"
    if hasattr(shard, "devices"):
        # PositionalSharding and others expose .devices()
        try:
            ndev = len(shard.devices())
        except Exception:
            ndev = "?"
        return f"{type(shard).__name__}(devices={ndev})"
    return str(shard)


def _get_array(obj):
    # nnx.Param-like leaves store the array in .value
    if hasattr(obj, "value") and hasattr(obj.value, "sharding"):
        return obj.value
    return obj


def _pytree_array_leaves(tree):
    leaves = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        arr = _get_array(leaf)
        if hasattr(arr, "shape") and hasattr(arr, "sharding"):
            leaves.append((path, arr))
    return leaves


def log_mesh_and_sharding_header(mesh: jax.sharding.Mesh, *, title: str):
    mesh_desc = ", ".join(f"{k}={v}" for k, v in mesh.shape.items())
    try:
        import numpy as _np

        total = int(_np.prod(list(mesh.shape.values())))
    except Exception:
        total = "?"
    logging.info(f"{title}: mesh axes [{mesh_desc}] total_devices={total}")


def log_batch_sharding(batch):
    def fmt_path(path):
        return jax.tree_util.keystr(path)

    lines = []
    for path, arr in _pytree_array_leaves(batch):
        try:
            ex_shape = None
            # Example addressable shard shape on this host (if available)
            if hasattr(arr, "addressable_shards") and arr.addressable_shards:
                ex_shape = arr.addressable_shards[0].data.shape
            shard_str = _format_sharding(arr.sharding)
            line = f"{fmt_path(path)}: global={tuple(arr.shape)} dtype={arr.dtype} | {shard_str}"
            if ex_shape is not None:
                line += f" | local_shard={tuple(ex_shape)}"
            lines.append(line)
        except Exception as e:
            lines.append(f"{fmt_path(path)}: <error formatting sharding: {e}>")
    if lines:
        logging.info("Batch sharding summary:\n" + "\n".join(lines))


def _decode_langact_strings(obs, tokenizer) -> list[str]:
    """Extract and decode the langact (language action) tokens per example."""
    return _decode_captions(obs, tokenizer)


def _decode_prompt_strings(obs, tokenizer) -> list[str]:
    """Extract and decode the prompt tokens per example."""
    if not hasattr(obs, "tokenized_prompt") or obs.tokenized_prompt is None:
        return []
    tokens = _safe_device_get(obs.tokenized_prompt)
    rmask = _safe_device_get(obs.tokenized_langact_mask)
    if tokens is None or rmask is None:
        return []
    out: list[str] = []
    for i in range(tokens.shape[0]):
        # Filter out padding tokens (typically 0)
        valid_tokens = tokens[i][~rmask[i].astype(bool)]
        try:
            text = tokenizer.decode(valid_tokens.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def _decode_captions(obs, tokenizer) -> list[str]:
    """Extract and decode the langact (language action) tokens per example."""
    if obs.tokenized_prompt is None or obs.tokenized_langact_mask is None:
        return []
    tokens = _safe_device_get(obs.tokenized_prompt)
    rmask = _safe_device_get(obs.tokenized_langact_mask)

    if tokens is None or rmask is None:
        return []
    out: list[str] = []
    for i in range(tokens.shape[0]):
        sel = tokens[i][rmask[i].astype(bool)]
        try:
            text = tokenizer.decode(sel.astype(np.int32))
        except Exception:
            text = ""
        out.append(text)
    return out


def _format_action_array(actions: np.ndarray, max_steps: int = 3) -> str:
    """Format action array as a human-readable string.

    Args:
        actions: Action array of shape [action_horizon, action_dim] or [action_dim]
        max_steps: Maximum number of timesteps to display

    Returns:
        Formatted string representation
    """
    if actions is None or actions.size == 0:
        return "No actions"

    # Handle both [H, D] and [D] shapes
    if len(actions.shape) == 1:
        actions = actions[None, :]

    horizon, action_dim = actions.shape

    # Format first few timesteps
    lines = []
    actions_sum = np.sum(actions[:, :7], axis=0)
    actions_sum[-1] = actions[-1, 6]
    # for t in range(min(max_steps, horizon)):
    #     # Show first 7 dims (xyz, rpy, gripper) for compactness
    #     act = actions[t, :7]
    #     line = f"t{t}: xyz=({act[0]:.2f},{act[1]:.2f},{act[2]:.2f}) rpy=({act[3]:.2f},{act[4]:.2f},{act[5]:.2f}) g={act[6]:.2f}"
    #     lines.append(line)

    act = actions_sum
    line = f"xyz=({act[0]:.2f},{act[1]:.2f},{act[2]:.2f}) rpy=({act[3]:.2f},{act[4]:.2f},{act[5]:.2f}) g={act[6]:.2f}"

    # if horizon > max_steps:
    #     lines.append(f"... ({horizon - max_steps} more steps)")

    # return " | ".join(lines)
    return line


def _extract_gt_actions(batch) -> list[str]:
    """Extract and format ground truth actions per example.

    Args:
        batch: Tuple of (observation, actions) from dataloader

    Returns:
        List of formatted action strings, one per batch element
    """
    if batch is None or len(batch) < 2:
        return []

    actions = _safe_device_get(batch[1])
    if actions is None:
        return []

    out: list[str] = []
    batch_size = actions.shape[0]

    for i in range(batch_size):
        # Extract actions for this example
        # Shape: [action_horizon, action_dim] or [action_dim]
        action_chunk = actions[i]
        formatted = _format_action_array(action_chunk, max_steps=2)
        out.append(formatted)

    return out


def _ensure_color(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=-1)
    if img.ndim != 3:
        return img
    if img.shape[-1] == 3:
        return img
    if img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    return img


def _wrap_text_to_lines(text: str, max_chars_per_line: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars_per_line:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines[:4]  # cap lines to avoid huge overlays


def _draw_text_block(img: np.ndarray, text: str, area: tuple[int, int, int, int]) -> np.ndarray:
    """Draw wrapped text with outline over a semi-transparent dark box.

    area: (x0, y0, x1, y1) in image coordinates.
    """
    try:
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont
    except Exception:
        return img
    x0, y0, x1, y1 = area
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)

    # Convert to PIL Image
    pil_img = Image.fromarray(img)

    # Create semi-transparent overlay
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 128))  # 50% alpha

    # Composite overlay onto image
    pil_img = pil_img.convert("RGBA")
    pil_img = Image.alpha_composite(pil_img, overlay)
    pil_img = pil_img.convert("RGB")

    # Text parameters scaled by height
    block_h = max(1, y1 - y0)
    base_scale = 2.5
    scale = max(0.4, min(1.5, block_h / 110.0)) * base_scale
    font_size = int(13 * scale * 0.4)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_img)
    # Calculate max chars based on available width
    available_width = x1 - x0 - 20  # subtract padding
    avg_char_width = font_size * 0.6  # rough estimate
    max_chars = max(20, int(available_width / avg_char_width))
    lines = _wrap_text_to_lines(text, max_chars)
    line_h = max(12, int(14 * scale))
    y = y0 + 8

    for line in lines:
        # Draw outline (black)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x0 + 8 + dx, y + dy), line, font=font, fill=(0, 0, 0))
        # Draw text (white)
        draw.text((x0 + 8, y), line, font=font, fill=(255, 255, 255))
        y += line_h - 4

    return np.array(pil_img)


def _make_legend_bar(width: int, height: int = 28) -> np.ndarray:
    try:
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont
    except Exception:
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        bar[:] = 32  # dark gray
        return bar

    # Create dark gray background
    bar = Image.new("RGB", (width, height), color=(32, 32, 32))
    draw = ImageDraw.Draw(bar)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            font = ImageFont.load_default()

    cx = 12
    items = [((0, 255, 255), "GT start"), ((0, 0, 255), "Pred end"), ((0, 255, 0), "GT end")]

    try:
        for color, label in items:
            # Draw circle (using ellipse with same width/height)
            radius = 6
            draw.ellipse([cx - radius, height // 2 - radius, cx + radius, height // 2 + radius], fill=color)
            # Draw text
            draw.text((cx + 12, height // 2 - 6), label, font=font, fill=(255, 255, 255))
            cx += 110
    except Exception:
        pass

    return np.array(bar)


def _compose_pages(rows: list[np.ndarray], target_max_height: int = 1600) -> list[np.ndarray]:
    pages: list[np.ndarray] = []
    if not rows:
        return pages
    row_h = rows[0].shape[0]
    legend_height = 40
    bottom_padding = 20  # Add padding at bottom so last text bar is visible
    per_page = max(1, (target_max_height - legend_height - bottom_padding) // row_h)
    for i in range(0, len(rows), per_page):
        chunk = rows[i : i + per_page]
        grid = np.concatenate(chunk, axis=0)
        legend = _make_legend_bar(grid.shape[1], height=legend_height)
        # Add bottom padding bar to prevent text cutoff
        padding_bar = np.zeros((bottom_padding, grid.shape[1], 3), dtype=np.uint8)
        padding_bar[:] = 32  # dark gray matching legend
        page = np.concatenate([legend, grid, padding_bar], axis=0)
        pages.append(page)
    return pages


def _is_tpu_runtime() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def main(config: _config.TrainConfig):
    wandb_enabled = bool(getattr(config, "wandb_enabled", False)) and wandb is not None
    if not wandb_enabled:
        if bool(getattr(config, "wandb_enabled", False)) and wandb is None:
            logging.warning("wandb requested but not installed; falling back to local image dumps.")
    else:
        wandb_mode = "online" if os.environ.get("WANDB_DISABLED", "false").lower() not in {"1", "true"} else "offline"
        run_name = f"vis-{config.name}"
        exp_name = getattr(config, "exp_name", None)
        if exp_name:
            run_name = f"{run_name}-{exp_name}"
        wandb_config = {
            "config_name": config.name,
            "exp_name": exp_name,
            "project_name": getattr(config, "project_name", "openpi-cot"),
        }
        wandb.init(
            project=getattr(config, "project_name", "openpi-cot"),
            name=run_name,
            config=wandb_config,
            reinit=True,
            mode=wandb_mode,
        )
        if hasattr(wandb.run, "log_code"):
            wandb.run.log_code(str(epath.Path(__file__).parent.parent))
    if ("v6" in config.name and config.fsdp_devices > 8) or ("v4" in config.name and config.fsdp_devices > 4):
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
    init_logging()
    logging.info(f"Local devices: {local_devices}, Global devices: {global_devices}, Process count: {process_count}")
    if process_count == 1:
        # Choose the largest divisor of available devices not exceeding configured fsdp_devices
        target = min(config.fsdp_devices, max(1, local_devices))
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
        assert global_devices % effective_fsdp_devices == 0

    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(effective_fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Human-readable mesh overview
    log_mesh_and_sharding_header(mesh, title="Device mesh")
    logging.info("Data sharding spec: %s", _format_sharding(data_sharding))

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    tok = PaligemmaTokenizer(max_len=300)

    data_iter = iter(data_loader)
    logging.info("Before getting batch")
    batch = next(data_iter)
    logging.info("After getting batch")
    # Sharding details for the first batch
    log_batch_sharding(batch)

    for j in range(10):
        # Visualize language-action projection per example
        obs = batch[0]
        # Decode langact and prompt strings
        langact_texts = _decode_langact_strings(obs, tok)
        prompt_texts = _decode_prompt_strings(obs, tok)
        # Extract captions (for VQA datasets)
        caption_texts = _decode_captions(obs, tok)
        # Extract ground truth actions
        gt_action_texts = _extract_gt_actions(batch)

        start_imgs = _safe_device_get(obs.images["base_0_rgb"])
        end_imgs = _safe_device_get(obs.images["left_wrist_0_rgb"])

        B = start_imgs.shape[0]
        vis_rows = []
        for i in range(B):
            start_u8 = np.asarray(((start_imgs[i] + 1.0) * 0.5 * 255.0).clip(0, 255), dtype=np.uint8)
            end_u8 = np.asarray(((end_imgs[i] + 1.0) * 0.5 * 255.0).clip(0, 255), dtype=np.uint8)
            la_text = langact_texts[i] if i < len(langact_texts) else ""
            prompt_text = prompt_texts[i] if i < len(prompt_texts) else ""
            caption_text = caption_texts[i] if i < len(caption_texts) else ""
            gt_action_text = gt_action_texts[i] if i < len(gt_action_texts) else ""

            # Determine if this is a VQA sample (has non-empty caption)
            is_vqa = bool(caption_text and caption_text.strip())
            is_vqa = obs.is_vqa_sample[i]

            # Combine prompt, langact/caption, and GT actions for display
            text_parts = []
            if prompt_text:
                text_parts.append(f"Prompt: {prompt_text}")

            # For VQA datasets, show caption instead of language actions
            if is_vqa:
                text_parts.append(f"Caption: {caption_text}")
            elif la_text:
                text_parts.append(f"LangAct: {la_text}")

            # Only show GT actions for robot datasets (not VQA)
            if gt_action_text and not is_vqa:
                text_parts.append(f"GT Action: {gt_action_text}")

            combined_text = " | ".join(text_parts)

            if is_vqa:
                logging.info(f"[{i}] [VQA] Prompt: {prompt_text} | Caption: {caption_text}")
            else:
                logging.info(f"[{i}] [Robot] Prompt: {prompt_text} | LangAct: {la_text}")
                logging.info(f"[{i}] GT Action: {gt_action_text}")

            col1 = np.copy(_ensure_color(start_u8))
            col2 = np.copy(_ensure_color(end_u8))
            panels = [col1]
            panels.append(col2)
            panels = [p for p in panels if p is not None]
            if not panels:
                continue
            row = np.concatenate(panels, axis=1)
            # Larger bottom overlay to fit action comparisons
            band_h_row = max(90, row.shape[0] // 5)
            row = _draw_text_block(
                row, combined_text, (4, row.shape[0] - band_h_row - 2, row.shape[1] - 4, row.shape[0] - 2)
            )
            vis_rows.append(row)
        if vis_rows:
            pages = _compose_pages(vis_rows, target_max_height=1600)
            assert wandb_enabled and wandb is not None
            wandb.log(
                {
                    "vis_dataset/prediction_images": [
                        wandb.Image(page, caption=f"batch_{j}_page_{pi:02d}") for pi, page in enumerate(pages)
                    ]
                },
                step=j,
            )

        batch = next(data_iter)

    if wandb_enabled and wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.finish()


if __name__ == "__main__":
    main(_config.cli())
