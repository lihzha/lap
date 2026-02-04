import dataclasses
from typing import TypeAlias

import numpy as np
from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
from openpi.transforms import DataTransformFn
from openpi.transforms import _assert_quantile_stats
from openpi.transforms import apply_tree
from openpi.transforms import flatten_dict
from openpi.transforms import unflatten_dict

from lap.datasets.utils.helpers import NormalizationType
from lap.models.tokenizer import FASTTokenizer
from lap.models.tokenizer import PaligemmaTokenizer

# Optional TF import: used to ensure ops run inside tf.data pipelines
try:  # pragma: no cover - optional dependency in some environments
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None

DataDict: TypeAlias = at.PyTree


@dataclasses.dataclass(frozen=True)
class TokenizePromptAndReasoning(DataTransformFn):
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = False
    dataset_name_pad_len: int = 100
    verbose_mode: bool = False
    state_dropout: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        # Always tokenize regular reasoning (prompt + language_actions)
        language_actions = data.pop("language_actions", None)  # if None, inference
        dataset_name = data.pop("dataset_name", None)  # if None, inference
        frame_description = data.pop("frame_description", "robot base frame")
        if dataset_name is not None:
            tokenized_dataset_name = self.tokenizer._tokenizer.encode(dataset_name)
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * (
                self.dataset_name_pad_len - len(tokenized_dataset_name)
            ) + tokenized_dataset_name
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)
        else:
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * self.dataset_name_pad_len
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)

        is_vqa_sample = data["is_vqa_sample"]
        is_prediction_sample = data["is_prediction_sample"]
        time_horizon_seconds = data.pop("time_horizon_seconds", None)

        # frame_description = "robot base frame"

        # Tokenize regular reasoning
        (
            tokens,
            pad_mask,
            reasoning_mask,
            numeric_mask,
            direction_mask,
            token_loss_mask,
        ) = self.tokenizer.tokenize(
            prompt,
            language_actions,
            state,
            is_vqa_sample=is_vqa_sample,
            is_prediction_sample=is_prediction_sample,
            time_horizon_seconds=time_horizon_seconds,
            frame_description=frame_description,
            state_dropout=self.state_dropout,
        )

        # Combine number_mask and direction_mask for critical tokens
        critical_mask = np.logical_or(numeric_mask, direction_mask)

        result = {
            **data,
            "tokenized_prompt": tokens,  # kept for compatibility with upstream
            "tokenized_prompt_mask": pad_mask,  # kept for compatibility with upstream
            "tokenized_langact_mask": reasoning_mask,
            "token_loss_mask": token_loss_mask,
            "tokenized_dataset_name": tokenized_dataset_name,
        }

        if self.verbose_mode:
            result.update(
                {  # Critical tokens are both numbers AND directional indicators
                    "critical_token_mask": critical_mask,
                    "number_token_mask": numeric_mask,
                    "direction_token_mask": direction_mask,
                }
            )

        return result


@dataclasses.dataclass(frozen=True)
class DetokenizeReasoning(DataTransformFn):
    tokenizer: PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if "tokens" in data:
            text = self.tokenizer.decode(data["tokens"].squeeze().astype(np.int32))
            return {**data, "reasoning": text}
        return data


@dataclasses.dataclass(frozen=True)
class SafeRepackTransform:
    structure: at.PyTree
    strict: bool = False  # when True, error if any source path is missing

    def __call__(self, data: DataDict) -> DataDict:
        flat_data = flatten_dict(data)
        flat_struct = flatten_dict(self.structure)  # maps out_key -> source_path or list of candidates
        out = {}
        missing = []
        for out_key, src_spec in flat_struct.items():
            # Allow a single source path or a list/tuple of fallback source paths.
            candidates = src_spec if isinstance(src_spec, (list, tuple)) else [src_spec]
            found = False
            for src_path in candidates:
                if src_path in flat_data:
                    out[out_key] = flat_data[src_path]
                    found = True
                    break
            if not found:
                missing.append((out_key, tuple(candidates)))
        if self.strict and missing:
            raise KeyError(f"Missing source paths: {missing}")
        return unflatten_dict(out)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # Normalization type to use (NORMAL, BOUNDS, or BOUNDS_Q99)
    normalization_type: NormalizationType | str = NormalizationType.NORMAL
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        if self.norm_stats is not None and normalization_type in (NormalizationType.BOUNDS_Q99,):
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        if normalization_type == NormalizationType.NORMAL:
            normalize_fn = self._normalize
        elif normalization_type == NormalizationType.BOUNDS:
            normalize_fn = self._normalize_bounds
        elif normalization_type == NormalizationType.BOUNDS_Q99:
            normalize_fn = self._normalize_quantile
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        return apply_tree(
            data,
            self.norm_stats,
            normalize_fn,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_bounds(self, x, stats: NormStats):
        assert stats.min is not None
        assert stats.max is not None
        min_val, max_val = stats.min[..., : x.shape[-1]], stats.max[..., : x.shape[-1]]
        # Scale to [-1, 1] and clip
        scaled = 2.0 * (x - min_val) / (max_val - min_val + 1e-8) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)
        # Zero-out dimensions with zero range
        zeros_mask = np.equal(min_val, max_val)
        while zeros_mask.ndim < x.ndim:
            zeros_mask = zeros_mask[None, ...]
        return np.where(zeros_mask, 0.0, scaled)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        # Scale to [-1, 1] and clip
        scaled = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        # scaled = np.clip(scaled, -1.0, 1.0)
        # Zero-out dimensions with zero range
        zeros_mask = np.equal(q01, q99)
        while zeros_mask.ndim < x.ndim:
            zeros_mask = zeros_mask[None, ...]
        return np.where(zeros_mask, 0.0, scaled)


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # Normalization type to use (NORMAL, BOUNDS, or BOUNDS_Q99)
    normalization_type: NormalizationType | str = NormalizationType.NORMAL

    def __post_init__(self):
        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        if self.norm_stats is not None and normalization_type in (NormalizationType.BOUNDS_Q99,):
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        if normalization_type == NormalizationType.NORMAL:
            unnormalize_fn = self._unnormalize
        elif normalization_type == NormalizationType.BOUNDS:
            unnormalize_fn = self._unnormalize_bounds
        elif normalization_type == NormalizationType.BOUNDS_Q99:
            unnormalize_fn = self._unnormalize_quantile
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            unnormalize_fn,
            strict=False,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_bounds(self, x, stats: NormStats):
        assert stats.min is not None
        assert stats.max is not None
        min_val = pad_to_dim(stats.min, x.shape[-1], axis=-1, value=-1.0)
        max_val = pad_to_dim(stats.max, x.shape[-1], axis=-1, value=1.0)
        return (x + 1.0) / 2.0 * (max_val - min_val + 1e-8) + min_val

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class PadStates(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class NormalizeActionAndProprio(DataTransformFn):
    """Normalize `action` and `proprio`-like fields using dataset statistics.

    This class adapts the behavior of `normalize_action_and_proprio` from
    `lap.datasets.oxe_utils.data_utils` into a DataTransformFn so it can
    be used directly in dataset pipelines.

    It supports both NumPy arrays and TensorFlow tensors. When TensorFlow tensors
    are encountered, TensorFlow math/clip/where ops are used to preserve graph
    execution; otherwise NumPy is used.
    """

    norm_stats: at.PyTree | None
    normalization_type: NormalizationType | str = NormalizationType.NORMAL
    action_key: str = "action"
    state_key: str = "proprio"

    def __call__(self, traj: DataDict) -> DataDict:
        if self.norm_stats is None:
            return traj

        normalization_type = self.normalization_type
        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)

        def _is_tf_tensor(x):
            return tf is not None and isinstance(x, tf.Tensor)

        def _to_tensor(value, like):
            if value is None:
                return None
            if _is_tf_tensor(like):
                if _is_tf_tensor(value):
                    return tf.cast(value, tf.float32)
                return tf.convert_to_tensor(value, dtype=tf.float32)
            # NumPy path
            if isinstance(value, np.ndarray):
                return value.astype(np.float32, copy=False)
            return np.asarray(value, dtype=np.float32)

        def _clip(x, lo, hi):
            if _is_tf_tensor(x):
                return tf.clip_by_value(x, lo, hi)
            return np.clip(x, lo, hi)

        def _where(mask, a, b):
            if _is_tf_tensor(a) or _is_tf_tensor(b):
                return tf.where(mask, a, b)
            return np.where(mask, a, b)

        def _equal(a, b):
            if _is_tf_tensor(a) or _is_tf_tensor(b):
                return tf.equal(a, b)
            return np.equal(a, b)

        def _get_group(stats_root, group_name: str):
            if isinstance(stats_root, dict):
                group = stats_root.get(group_name)
                if group is None and group_name.endswith("s"):
                    group = stats_root.get(group_name[:-1])
                return group
            return None

        def _get_value(group_stats, key: str, like_arr):
            if group_stats is None:
                return None
            if isinstance(group_stats, dict):
                value = group_stats.get(key)
            else:
                value = getattr(group_stats, key, None)
            if value is None:
                return None
            return _to_tensor(value, like_arr)

        # Resolve stats groups
        actions_stats = _get_group(self.norm_stats, "actions")
        state_stats = _get_group(self.norm_stats, "state")

        # Fetch action/state tensors from the trajectory and harmonize dtypes
        action_arr = traj[self.action_key]
        state_container = traj.get("observation", {})
        state_arr = state_container.get(self.state_key)

        # Ensure numeric dtype consistency (float32) before arithmetic to avoid TF float64 vs float32 issues
        if _is_tf_tensor(action_arr):
            action_arr = tf.cast(action_arr, tf.float32)
        else:
            action_arr = np.asarray(action_arr, dtype=np.float32)

        if state_arr is not None:
            if _is_tf_tensor(state_arr):
                state_arr = tf.cast(state_arr, tf.float32)
            else:
                state_arr = np.asarray(state_arr, dtype=np.float32)
            # Update the container to keep downstream consistency
            traj.setdefault("observation", {})[self.state_key] = state_arr
        # Update the action in traj to the standardized dtype pre-normalization
        traj[self.action_key] = action_arr

        if normalization_type == NormalizationType.NORMAL:
            a_mean = _get_value(actions_stats, "mean", action_arr)
            a_std = _get_value(actions_stats, "std", action_arr)
            s_mean = _get_value(state_stats, "mean", state_arr) if state_arr is not None else None
            s_std = _get_value(state_stats, "std", state_arr) if state_arr is not None else None

            if a_mean is not None and a_std is not None:
                if _is_tf_tensor(action_arr):
                    traj[self.action_key] = (action_arr - a_mean) / (a_std + 1e-6)
                else:
                    traj[self.action_key] = (action_arr - a_mean) / (a_std + 1e-6)

            if state_arr is not None and s_mean is not None and s_std is not None:
                normed = (state_arr - s_mean) / (s_std + 1e-6)
                traj["observation"][self.state_key] = normed

        elif normalization_type in (NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99):
            low_key = "min" if normalization_type == NormalizationType.BOUNDS else "q01"
            high_key = "max" if normalization_type == NormalizationType.BOUNDS else "q99"

            a_low = _get_value(actions_stats, low_key, action_arr)
            a_high = _get_value(actions_stats, high_key, action_arr)
            s_low = _get_value(state_stats, low_key, state_arr) if state_arr is not None else None
            s_high = _get_value(state_stats, high_key, state_arr) if state_arr is not None else None

            if a_low is not None and a_high is not None:
                # scale to [-1, 1] and clip
                scaled = 2.0 * (action_arr - a_low) / (a_high - a_low + (1e-8)) - 1.0
                traj[self.action_key] = _clip(scaled, -1.0, 1.0)
                # zero-out dimensions with zero range
                zeros_mask = _equal(a_low, a_high)  # shape [..., D]
                # Broadcast mask over leading dims of action tensor
                if _is_tf_tensor(action_arr):
                    while len(zeros_mask.shape) < len(action_arr.shape):
                        zeros_mask = zeros_mask[None, ...]
                else:
                    while zeros_mask.ndim < action_arr.ndim:
                        zeros_mask = zeros_mask[None, ...]
                traj[self.action_key] = _where(zeros_mask, 0.0, traj[self.action_key])

            if state_arr is not None and s_low is not None and s_high is not None:
                scaled = 2.0 * (state_arr - s_low) / (s_high - s_low + (1e-8)) - 1.0
                state_normed = _clip(scaled, -1.0, 1.0)
                zeros_mask = _equal(s_low, s_high)
                if _is_tf_tensor(state_normed):
                    while len(zeros_mask.shape) < len(state_normed.shape):
                        zeros_mask = zeros_mask[None, ...]
                else:
                    while zeros_mask.ndim < state_normed.ndim:
                        zeros_mask = zeros_mask[None, ...]
                traj["observation"][self.state_key] = _where(zeros_mask, 0.0, state_normed)

        return traj


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    """Tokenize inputs for FAST model.

    This combines CoT-style reasoning with FAST-style action token prediction:
    - Tokenizes prompt + language actions (if available)
    - Appends state and action representations as tokens
    - Creates appropriate masks for training

    Similar to upstream TokenizeFASTInputs but with support for language actions.
    For VQA and prediction samples, includes the language_actions (answer) tokens
    in the loss mask so they contribute to training.
    """

    tokenizer: FASTTokenizer
    discrete_state_input: bool = True
    state_dropout: float = 0.0
    dataset_name_pad_len: int = 100

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = data.get("state")
            if state is None:
                raise ValueError("State is required for FAST tokenization.")

        time_horizon_seconds = data.pop("time_horizon_seconds", None)
        frame_description = data.pop("frame_description", "robot base frame")

        # Get language_actions (contains VQA answer or prediction answer)
        # Similar to TokenizePromptAndReasoning pattern
        language_actions = data.pop("language_actions", None)

        dataset_name = data.pop("dataset_name", None)  # if None, inference
        if dataset_name is not None:
            tokenized_dataset_name = self.tokenizer._tokenizer.encode(dataset_name)
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * (
                self.dataset_name_pad_len - len(tokenized_dataset_name)
            ) + tokenized_dataset_name
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)
        else:
            pad_id = self.tokenizer._tokenizer.pad_id()
            tokenized_dataset_name = [pad_id] * self.dataset_name_pad_len
            tokenized_dataset_name = np.asarray(tokenized_dataset_name, dtype=np.int32)

        # Get state type if available
        state_type = data.pop("state_type", None)
        if state_type is not None and not isinstance(state_type, str):
            state_type = state_type.item() if hasattr(state_type, "item") else str(state_type)

        # Get VQA and prediction flags
        is_vqa_sample = data.get("is_vqa_sample", False)
        is_prediction_sample = data.get("is_prediction_sample", False)

        # Get actions (None during inference)
        actions = data.get("actions")

        # Tokenize using the FAST tokenizer
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize_fast(
            prompt=prompt,
            state=state,
            actions=actions,
            language_actions=language_actions,
            state_type=state_type,
            is_vqa_sample=is_vqa_sample,
            is_prediction_sample=is_prediction_sample,
            time_horizon_seconds=time_horizon_seconds,
            state_dropout=self.state_dropout,
            frame_description=frame_description,
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "tokenized_langact_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "tokenized_dataset_name": tokenized_dataset_name,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "tokens" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("tokens")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    x = x[..., :target_dim]  # Truncate if necessary
    return x
