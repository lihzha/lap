from collections import defaultdict
import dataclasses
import logging
import pathlib
import re
from typing import Literal, Protocol, runtime_checkable

from flax import traverse_util
import flax.traverse_util
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.shared.array_typing as at
import orbax.checkpoint as ocp
from scipy import ndimage

import lap.shared.download as download

logger = logging.getLogger(__name__)


def recover_dtype(a: np.ndarray) -> np.ndarray:
    """Numpy's `save` stores bfloat16 type as "void" type, so we recover it."""
    if hasattr(a, "dtype") and a.dtype.type is np.void:
        assert a.itemsize == 2, "Unknown dtype!"
        return a.view(jax.numpy.bfloat16)
    return a


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        # Preferred logic: use remote cache if present, otherwise mirror upstream into cache and load from there.
        params_path_str = str(self.params_path)

        if params_path_str.startswith("gs://"):
            # If this is already a cache path, try it; if missing or incomplete, fall back to upstream and mirror.
            if "/cache/" in params_path_str:
                cache_candidate = params_path_str
                upstream = params_path_str.split("/cache/", 1)[1]
                upstream = upstream if upstream.startswith("gs://") else f"gs://{upstream}"
                # Prefer existing cache; if present, ensure commit_success marker and use it.
                # Otherwise, mirror upstream into cache and use the mirror.
                try:
                    download.ensure_commit_success(cache_candidate)
                    params_source = cache_candidate
                except Exception:
                    params_source = str(download.maybe_download(upstream))
            else:
                # Not in cache yet; mirror upstream into cache to standardize layout.
                params_source = str(download.maybe_download(params_path_str))
        else:
            params_source = str(download.maybe_download(params_path_str))

        # def get_all_keys(d, prefix=""):
        #     keys = []
        #     for k, v in d.items():
        #         full_key = f"{prefix}.{k}" if prefix else k
        #         keys.append(full_key)
        #         if isinstance(v, dict):
        #             keys.extend(get_all_keys(v, prefix=full_key))
        #     return keys

        # all_keys = get_all_keys(params)
        # print(all_keys)

        loaded_params = _model.restore_params(params_source, restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PaliGemma2WeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma2 checkpoint."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(self.params_path, gs={"token": "anon"})
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        loaded_params = jax.tree.map(recover_dtype, loaded_params)
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        # params = ckptr.restore(params_path,ocp.args.PyTreeRestore(item=metadata,restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=None, restore_type=np.ndarray, dtype=None), metadata),),)

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), metadata
                ),
            ),
        )

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def print_param_shapes(d, path=""):
    """
    Recursively traverses a nested dictionary and prints the path and shape
    of any value that has a 'shape' attribute.

    Args:
      d (dict): The dictionary to traverse.
      path (str): The current key path (used for recursion).
    """
    for key, value in d.items():
        # Append the current key to the path
        new_path = f"{path}['{key}']"

        if isinstance(value, dict):
            # If the value is another dictionary, recurse deeper
            print_param_shapes(value, new_path)
        elif hasattr(value, "shape"):
            # If the value has a 'shape' attribute, print the path and shape
            print(f"Param: {new_path} | Shape: {value.shape}")


def get_param_info(params: dict) -> dict:
    """
    Flattens a nested dictionary of parameters and returns a dictionary
    mapping each parameter's path (as a string) to its shape and size.
    """
    # Using '/' as a separator is standard and simpler for set operations.
    flat_params = flatten_dict(params, sep="/")
    info = {}
    for key, value in flat_params.items():
        if hasattr(value, "shape") and hasattr(value, "size"):
            info[key] = {"shape": value.shape, "size": value.size}
    return info


def compare_checkpoints(source_info: dict, target_info: dict):
    """
    Compares two parameter dictionaries (source and target) to debug loading.

    - Checks for parameter names in source that are missing in target.
    - Checks for shape mismatches for common parameters.
    - Lists parameters in target that are not in source (uninitialized).
    """
    source_keys = set(source_info.keys())
    target_keys = set(target_info.keys())

    print("\n--- Starting Checkpoint Comparison ---")

    # Critical Error: Keys in our remapped checkpoint that DON'T exist in the model
    mismatched_keys = source_keys - target_keys
    if mismatched_keys:
        print(
            f"\n❌ ERROR: Found {len(mismatched_keys)} keys in the remapped checkpoint that are NOT in the final model:"
        )
        for key in sorted(list(mismatched_keys)):
            print(f"  - {key}")
    else:
        print("\n✅ SUCCESS: All remapped checkpoint keys exist in the final model.")

    # Informational: Keys in the model that are NOT loaded from the checkpoint
    uninitialized_keys = target_keys - source_keys
    if uninitialized_keys:
        print(
            f"\nℹ️ INFO: Found {len(uninitialized_keys)} keys in the final model that will NOT be loaded from the checkpoint (expected for new layers):"
        )
        for key in sorted(list(uninitialized_keys)):
            print(f"  - {key} (Shape: {target_info[key]['shape']})")

    # Critical Error: Shape mismatches for keys that exist in both
    shape_mismatches = []
    common_keys = source_keys.intersection(target_keys)
    for key in common_keys:
        source_shape = source_info[key]["shape"]
        target_shape = target_info[key]["shape"]
        if source_shape != target_shape:
            shape_mismatches.append((key, source_shape, target_shape))

    if shape_mismatches:
        print(f"\n❌ ERROR: Found {len(shape_mismatches)} shape mismatches between the checkpoint and the model:")
        for key, source_shape, target_shape in shape_mismatches:
            print(f"  - Key: {key}")
            print(f"    Checkpoint Shape: {source_shape} -> Model Shape: {target_shape}")
    else:
        print("\n✅ SUCCESS: All common parameter shapes match.")

    print("\n--- End of Checkpoint Comparison ---\n")


@dataclasses.dataclass(frozen=True)
class Gemma3ScanCompatibleWeightLoader(WeightLoader):
    """Loads and remaps Gemma3 weights to match Pi0's nn.scan naming conventions.

    This loader:
    1. Loads raw Gemma3 checkpoint with per-layer naming (layer_0, layer_1, ...)
    2. Stacks per-layer weights into a single 'layers' array dimension
    3. Remaps key names (_key_norm -> k_rmsnorm, gating_einsum -> Einsum_0, etc.)
    4. Extracts and remaps vision encoder (SigLiP) from per-layer to stacked format
    5. Extracts embedder to PaliGemma namespace
    6. Optionally resizes SigLiP positional embeddings to match target patch count
    """

    params_path: str
    target_pos_emb_grid_size: tuple[int, int] | None = (16, 16)  # e.g., (16, 16) for 256 patches

    def _resize_positional_embedding(
        self, pos_emb: np.ndarray, original_grid_size: tuple[int, int], target_grid_size: tuple[int, int]
    ) -> np.ndarray:
        """Resize 2D positional embeddings using bicubic interpolation.

        Args:
            pos_emb: Positional embedding array of shape [1, orig_h*orig_w, dim]
            original_grid_size: (orig_h, orig_w) e.g., (64, 64) for 4096 patches
            target_grid_size: (new_h, new_w) e.g., (16, 16) for 256 patches

        Returns:
            Resized positional embedding of shape [1, new_h*new_w, dim]
        """
        orig_h, orig_w = original_grid_size
        new_h, new_w = target_grid_size
        dim = pos_emb.shape[-1]

        logger.info(
            f"Resizing positional embeddings from {orig_h}x{orig_w} ({orig_h * orig_w} patches) "
            f"to {new_h}x{new_w} ({new_h * new_w} patches)"
        )

        # Ensure input is a proper numpy array and preserve dtype
        original_dtype = pos_emb.dtype
        if not isinstance(pos_emb, np.ndarray):
            pos_emb = np.asarray(pos_emb)

        # Reshape to 2D grid: [1, H*W, D] -> [1, H, W, D]
        pos_emb_2d = pos_emb.reshape(1, orig_h, orig_w, dim)

        # Use scipy.ndimage.zoom for bicubic interpolation (pure numpy, no device issues)
        # Compute zoom factors for each dimension: [batch, height, width, channels]
        zoom_factors = [1.0, new_h / orig_h, new_w / orig_w, 1.0]

        # order=3 gives bicubic interpolation
        pos_emb_resized = ndimage.zoom(pos_emb_2d, zoom_factors, order=3, mode="reflect")

        # Reshape back to sequence: [1, H, W, D] -> [1, H*W, D]
        result = pos_emb_resized.reshape(1, new_h * new_w, dim).astype(original_dtype)

        # Verify it's a numpy array
        assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"

        return result

    def _remap_siglip(self, siglip_params: dict, target_pos_emb_grid_size: tuple[int, int] | None = None) -> dict:
        """Remap SigLiP from encoderblock_0, encoderblock_1, ... to stacked encoderblock.

        Args:
            siglip_params: Raw SigLiP parameters from checkpoint
            target_pos_emb_grid_size: Target grid size for positional embeddings (h, w).
                                     If None, no resizing will be performed.
        """
        siglip_encoder = siglip_params.get("siglip_encoder", {})
        transformer = siglip_encoder.get("Transformer", {})

        # Pattern to match encoderblock_0, encoderblock_1, etc.
        encoderblock_pattern = re.compile(r"encoderblock_(\d+)")

        # Separate encoder blocks from other components
        encoder_blocks = {}
        other_components = {}

        for key, value in transformer.items():
            m = encoderblock_pattern.match(key)
            if m:
                layer_idx = int(m.group(1))
                encoder_blocks[layer_idx] = value
            else:
                other_components[key] = value

        # Stack encoder blocks
        if encoder_blocks:
            # Get the structure from first block
            num_layers = max(encoder_blocks.keys()) + 1
            first_block = encoder_blocks[0]

            # Flatten each block and collect by subkey
            stacked_weights = defaultdict(list)
            for layer_idx in range(num_layers):
                block = encoder_blocks[layer_idx]
                flat_block = flax.traverse_util.flatten_dict(block, sep="/")
                for subkey, subval in flat_block.items():
                    stacked_weights[subkey].append((layer_idx, subval))

            # Stack into single arrays
            stacked_block = {}
            for subkey, layer_values in stacked_weights.items():
                layer_values.sort(key=lambda x: x[0])
                arrays = [v for _, v in layer_values]
                # Convert subkey string back to nested dict structure
                # Use np.stack to keep as numpy arrays (not JAX arrays)
                stacked_block[tuple(subkey.split("/"))] = np.stack(arrays, axis=0)

            # Unflatten the stacked block
            encoderblock_dict = flax.traverse_util.unflatten_dict(stacked_block)

            # Reconstruct transformer with stacked encoderblock
            new_transformer = {"encoderblock": encoderblock_dict, **other_components}
        else:
            new_transformer = other_components

        # Reconstruct full structure
        result = {
            "Transformer": new_transformer,
            "embedding": siglip_params.get("siglip_encoder", {}).get("embedding", siglip_encoder.get("embedding", {})),
        }

        # Add head and pos_embedding if they exist at siglip_encoder level
        if "head" in siglip_encoder:
            result["head"] = siglip_encoder["head"]
        if "pos_embedding" in siglip_encoder:
            pos_emb = siglip_encoder["pos_embedding"]

            # Resize positional embeddings if target grid size is specified
            if target_pos_emb_grid_size is not None:
                current_num_patches = pos_emb.shape[1]  # Shape is [1, num_patches, dim]
                current_grid_size = int(np.sqrt(current_num_patches))

                # Verify it's a square grid
                if current_grid_size * current_grid_size != current_num_patches:
                    logger.warning(
                        f"Non-square positional embedding grid detected: {current_num_patches} patches. "
                        f"Assuming grid size of {current_grid_size}x{current_grid_size}"
                    )

                target_h, target_w = target_pos_emb_grid_size

                # Only resize if sizes differ
                if current_grid_size != target_h or current_grid_size != target_w:
                    pos_emb = self._resize_positional_embedding(
                        pos_emb, (current_grid_size, current_grid_size), target_pos_emb_grid_size
                    )
                    logger.info(f"Positional embedding resized to: {target_h}x{target_w}")
                else:
                    logger.info(f"Positional embedding already at target size: {target_h}x{target_w}")

            result["pos_embedding"] = pos_emb

        return result

    def load(self, params: at.Params) -> at.Params:
        logger.info("Loading Gemma3 weights using Gemma3ScanCompatibleWeightLoader...")

        # Determine target positional embedding size from model params
        target_pos_emb_grid_size = self.target_pos_emb_grid_size
        if target_pos_emb_grid_size is None:
            # Try to auto-detect from model params
            try:
                model_pos_emb_shape = params.get("PaliGemma", {}).get("img", {}).get("pos_embedding", None)
                if model_pos_emb_shape is not None and hasattr(model_pos_emb_shape, "shape"):
                    target_num_patches = model_pos_emb_shape.shape[1]
                    target_grid_size = int(np.sqrt(target_num_patches))
                    if target_grid_size * target_grid_size == target_num_patches:
                        target_pos_emb_grid_size = (target_grid_size, target_grid_size)
                        logger.info(
                            f"Auto-detected target positional embedding grid size: "
                            f"{target_grid_size}x{target_grid_size} ({target_num_patches} patches)"
                        )
                    else:
                        logger.warning(f"Could not auto-detect square grid size from {target_num_patches} patches")
            except Exception as e:
                logger.warning(f"Could not auto-detect target positional embedding size: {e}")

        # Load raw checkpoint
        loaded_params = restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # print_param_shapes(loaded_params)
        # print("")
        # print_param_shapes(params)

        # Do everything on CPU to avoid device memory issues during remapping
        with jax.default_device(jax.devices("cpu")[0]):
            # Turn 'a/b/c' -> ('a','b','c') and unflatten
            flat_dict = {}
            for k, v in loaded_params.items():
                # If k is already a tuple, use it
                if isinstance(k, tuple):
                    flat_dict[k] = v
                # Otherwise split only once
                elif isinstance(k, str):
                    flat_dict[tuple(k.split("/"))] = v
                else:
                    raise TypeError(f"Unexpected key type: {type(k)} for key {k}")

            flat_original = flax.traverse_util.unflatten_dict(flat_dict)

            logger.info("Remapping checkpoint keys to match the nn.scan model structure...")

            # ===== TRANSFORMER (LLM) REMAPPING =====
            transformer = flat_original.get("transformer", {})
            layer_pattern = re.compile(r"layer_(\d+)")

            weights_to_stack = defaultdict(list)
            layerless = {}

            # Separate per-layer and global transformer weights
            for key, value in transformer.items():
                m = layer_pattern.match(key)
                if m:
                    layer_idx = int(m.group(1))
                    for subkey, subval in flax.traverse_util.flatten_dict(value, sep="/").items():
                        weights_to_stack[subkey].append((layer_idx, subval))
                else:
                    layerless[key] = value  # e.g., final_norm, embedder, etc.

            # Stack all layer weights
            flat_remapped = {}
            for subkey, layer_values in weights_to_stack.items():
                layer_values.sort(key=lambda x: x[0])
                arrays = [v for _, v in layer_values]
                # Use np.stack to keep as numpy arrays (not JAX arrays)
                flat_remapped[("layers",) + tuple(subkey.split("/"))] = np.stack(arrays, axis=0)

            # Add layerless (non-layer) weights back
            for key, value in flax.traverse_util.flatten_dict(layerless, sep="/").items():
                # Use np.asarray to keep as numpy arrays (not JAX arrays)
                flat_remapped[("transformer",) + tuple(key.split("/"))] = np.asarray(value)

            # Rebuild final structure
            remapped_params = flax.traverse_util.unflatten_dict(flat_remapped)

            # Do final adjustments to naming
            if "final_norm" in remapped_params.get("transformer", {}):
                remapped_params["final_norm"] = remapped_params["transformer"].pop("final_norm")

            if "embedder" in remapped_params.get("transformer", {}):
                remapped_params["embedder"] = remapped_params["transformer"].pop("embedder")

            # Change _key_norm -> k_rmsnorm and _query_norm -> q_rmsnorm
            remapped_params["layers"]["attn"]["k_rmsnorm"] = remapped_params["layers"]["attn"].pop("_key_norm")
            remapped_params["layers"]["attn"]["q_rmsnorm"] = remapped_params["layers"]["attn"].pop("_query_norm")

            # Fix Einsum in Gemma3
            gating_dict = remapped_params["layers"]["mlp"].pop("gating_einsum")
            remapped_params["layers"]["mlp"]["Einsum_0"] = {}
            remapped_params["layers"]["mlp"]["Einsum_0"]["gating_einsum"] = gating_dict["w"]

            linear_dict = remapped_params["layers"]["mlp"].pop("linear")
            remapped_params["layers"]["mlp"]["Einsum_1"] = {}
            remapped_params["layers"]["mlp"]["Einsum_1"]["linear"] = linear_dict["w"]

            # ===== SIGLIP (VISION ENCODER) REMAPPING =====
            siglip_remapped = None
            if "SigLiPFromPatches_0" in flat_original:
                logger.info("Remapping SigLiP vision encoder...")
                siglip_remapped = self._remap_siglip(
                    flat_original["SigLiPFromPatches_0"], target_pos_emb_grid_size=target_pos_emb_grid_size
                )

            # ===== BUILD PALIGEMMA STRUCTURE =====

            logger.info("Remapping and preparing parameters for final model structure...")

            # This will be our final, clean dictionary of remapped parameters to be loaded.
            flat_llm = {}

            # --- Part A: Handle Special Embedder Weights ---
            # We process these first by MOVING them out of the main `remapped_params` dict.
            # Using .pop() retrieves the value AND removes it, preventing duplication later.
            embedder_params = remapped_params.get("embedder", {})

            if "input_embedding" in embedder_params:
                flat_llm["PaliGemma/llm/embedder/input_embedding"] = embedder_params.pop("input_embedding")

            if "mm_input_projection" in embedder_params:
                # The debug output shows the final model needs this weight here:
                mm_projection = embedder_params.pop("mm_input_projection")  # <-- MODIFIED

                # Map the kernel
                flat_llm["PaliGemma/img/head/kernel"] = mm_projection["w"]

                # Map the bias, which is the missing parameter
                if "b" in mm_projection:  # Check if bias exists
                    print("Mapping mm_input_projection bias to PaliGemma/img/head/bias")
                    flat_llm["PaliGemma/img/head/bias"] = mm_projection["b"]  # <-- NEW LINE

            # We previously discarded this, but now we know the correct destination name
            if "mm_soft_embedding_norm" in embedder_params:
                flat_llm["PaliGemma/img/mm_soft_embedding_norm/scale"] = embedder_params.pop("mm_soft_embedding_norm")[
                    "scale"
                ]  # <-- MODIFIED

            # Now, put the modified (and smaller) embedder_params back, if anything is left.
            if embedder_params:
                remapped_params["embedder"] = embedder_params
            else:
                remapped_params.pop("embedder", None)

            # --- Part B: Handle SigLIP Vision Encoder Weights ---
            if siglip_remapped:
                flat_siglip = flax.traverse_util.flatten_dict(siglip_remapped, sep="/")
                flat_llm.update({f"PaliGemma/img/{k}": v for k, v in flat_siglip.items()})

            # --- Part C: Handle the Rest of the LLM Weights ---
            flat_remaining_llm = flax.traverse_util.flatten_dict(remapped_params, sep="/")
            flat_llm.update({f"PaliGemma/llm/{k}": v for k, v in flat_remaining_llm.items()})

            # ==========================================================
            # ===== RUN DEBUGGING CHECKS ON OUR CLEANED PARAMS =========
            # ==========================================================
            logger.info("Running post-remapping validation checks...")

            original_info = get_param_info(loaded_params)
            original_total_params = sum(p["size"] for p in original_info.values())

            final_llm_nested = unflatten_dict({tuple(k.split("/")): v for k, v in flat_llm.items()})
            final_llm_info = get_param_info(final_llm_nested)
            final_total_params = sum(p["size"] for p in final_llm_info.values())

            print("\n--- Starting Parameter Conservation Check ---")
            print(f"Total params in original checkpoint: {original_total_params:,}")
            print(f"Total params in final remapped dict: {final_total_params:,}")
            if original_total_params >= final_total_params:
                print(
                    f"✅ SUCCESS: Parameter count is valid. Discarded {original_total_params - final_total_params:,} parameters that are not in the target model."
                )
            else:
                print(
                    f"❌ ERROR: Parameter count mismatch! Gained {final_total_params - original_total_params:,} parameters, indicating duplication."
                )
            print("--- End of Conservation Check ---\n")

            final_model_info = get_param_info(params)
            compare_checkpoints(final_llm_info, final_model_info)
            # ==========================================================
            # ==========================================================

            # Now, with a clean `flat_llm`, we can perform the merge.
            flat_model = flax.traverse_util.flatten_dict(params, sep="/")
            merged = _merge_params(flat_llm, flat_model, missing_regex=".*")

        return merged


@dataclasses.dataclass(frozen=True)
class WeightLoaderChoice(WeightLoader):
    """CLI-friendly wrapper to choose a weight loader without nested subcommands.

    This class implements the WeightLoader protocol and forwards to a concrete
    loader based on the selected kind. It allows setting the loader type and its
    arguments via flat flags like:

      --weight-loader.kind=checkpoint --weight-loader.params-path=gs://...
      --weight-loader.kind=paligemma
      --weight-loader.kind=gemma3 --weight-loader.target-pos-emb-grid-size='(16,16)'
      --weight-loader.kind=pi05_action_expert --weight-loader.params-path=gs://base/checkpoint --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=pi05_base --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=paligemma_with_pi05_action_expert --weight-loader.params-path=gs://paligemma/checkpoint --weight-loader.pi05-params-path=gs://...
      --weight-loader.kind=none
    """

    # Which loader to use.
    kind: Literal[
        "none",
        "checkpoint",
        "paligemma",
        "paligemma2",
        "gemma3",
    ] = "paligemma"
    # Used when kind == "checkpoint", "paligemma2", "gemma3".
    params_path: str | None = None
    # Only used when kind == "gemma3" - target grid size for positional embeddings.
    target_pos_emb_grid_size: tuple[int, int] | None = None
    # Only used when kind == "pi05_action_expert", "pi05_base", or "paligemma_with_pi05_action_expert" - path to Pi0.5 checkpoint.
    pi05_params_path: str | None = "gs://openpi-assets/checkpoints/pi05_base/params"

    def _resolve(self) -> WeightLoader:
        match self.kind:
            case "checkpoint":
                if not self.params_path:
                    raise ValueError("--weight-loader.params-path must be set when kind=checkpoint")
                return CheckpointWeightLoader(self.params_path)
            case "paligemma":
                return PaliGemmaWeightLoader()
            case "paligemma2":
                if not self.params_path:
                    raise ValueError("--weight-loader.params-path must be set when kind=paligemma2")
                return PaliGemma2WeightLoader(self.params_path)
            case "gemma3":
                if not self.params_path:
                    raise ValueError("--weight-loader.params-path must be set when kind=gemma3")
                return Gemma3ScanCompatibleWeightLoader(
                    self.params_path, target_pos_emb_grid_size=self.target_pos_emb_grid_size
                )

            case "none":
                return NoOpWeightLoader()
            case _:
                raise ValueError(f"Unknown weight loader kind: {self.kind}")

    def load(self, params: at.Params) -> at.Params:
        return self._resolve().load(params)


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
