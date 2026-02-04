"""Output transformation utilities for CoT policy."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from lap.policies.lang_action_formats import get_language_action_format

if TYPE_CHECKING:
    from lap.policies.lang_action_formats import LanguageActionFormat


class OutputDecodingStrategy(Protocol):
    def decode(self, *, transform: CoTOutputs, data: dict, reasoning: str) -> dict: ...


@dataclasses.dataclass(frozen=True)
class CoTOutputs:
    """Transform for processing model outputs back to robot actions.

    Handles:
    - Parsing language action outputs to numeric deltas
    - VLA-0 format unnormalization
    - Action array construction

    Attributes:
        language_action_format: Format specification for parsing outputs.
        norm_stats: Normalization statistics for VLA-0 unnormalization.
        normalization_type: Type of normalization used ("bounds_q99", "bounds", "normal").
    """

    language_action_format: LanguageActionFormat | str | None = None
    norm_stats: dict | None = None
    normalization_type: str = "bounds_q99"
    transform_strategy: Literal["standard", "vla0"] = "standard"

    def __post_init__(self):
        """Resolve string schema name to LanguageActionFormat instance."""
        from lap.policies.lang_action_formats import LanguageActionFormat

        if self.language_action_format is not None and not isinstance(
            self.language_action_format, LanguageActionFormat
        ):
            schema = get_language_action_format(self.language_action_format)
            object.__setattr__(self, "language_action_format", schema)
        object.__setattr__(self, "_decoder", self._build_decoder())

    def _build_decoder(self) -> OutputDecodingStrategy:
        if self.transform_strategy == "vla0":
            return VLA0OutputDecoder()
        return StandardOutputDecoder()

    def __call__(self, data: dict) -> dict:
        """Process model output to extract actions.

        Args:
            data: Output dictionary containing 'actions' and optionally 'reasoning'.

        Returns:
            Dictionary with 'actions' array and 'reasoning' string.
        """
        if "reasoning" not in data:
            return {
                "actions": np.asarray(data["actions"][:, :7]),
                "reasoning": None,
            }

        reasoning = data.get("reasoning")
        assert self.language_action_format is not None
        assert reasoning is not None

        return self._decoder.decode(transform=self, data=data, reasoning=reasoning)

    def _process_standard_output(self, data: dict, reasoning: str) -> dict:
        """Process standard format output.

        Args:
            data: Output data dictionary.
            reasoning: Language action output string.

        Returns:
            Dictionary with parsed actions and reasoning.
        """
        # Extract initial state for EEF frame transformation
        initial_state = None
        if self.language_action_format.use_eef_frame and "raw_state" in data:
            initial_state = np.asarray(data["raw_state"])

        # Parse reasoning to deltas
        movement, gripper_action = self.language_action_format.parse_language_to_deltas(
            reasoning, initial_state=initial_state
        )

        # Construct action array
        if gripper_action is not None:
            single_action = np.concatenate([movement, [gripper_action]])
        else:
            single_action = movement

        return {"actions": single_action, "reasoning": reasoning}

    def _unnormalize_vla0_actions(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize VLA0 actions from [-1, 1] to physical space.

        Args:
            actions: Normalized action array.

        Returns:
            Unnormalized action array.
        """
        if self.norm_stats is None:
            return actions

        actions_stats = self.norm_stats.get("actions")
        if actions_stats is None:
            return actions

        if self.normalization_type == "bounds_q99":
            return self._unnormalize_bounds_q99(actions, actions_stats)
        if self.normalization_type == "bounds":
            return self._unnormalize_bounds(actions, actions_stats)
        if self.normalization_type == "normal":
            return self._unnormalize_normal(actions, actions_stats)

        return actions

    def _unnormalize_bounds_q99(self, actions: np.ndarray, stats) -> np.ndarray:
        """Unnormalize using q01/q99 bounds."""
        q01 = getattr(stats, "q01", None)
        q99 = getattr(stats, "q99", None)
        if q01 is None or q99 is None:
            return actions

        q01 = np.asarray(q01)
        q99 = np.asarray(q99)
        dim = min(q01.shape[-1], actions.shape[-1])

        # Unnormalize from [-1, 1] to original space
        unnormed = (actions[..., :dim] + 1.0) / 2.0 * (q99[..., :dim] - q01[..., :dim] + 1e-6) + q01[..., :dim]

        if actions.shape[-1] > dim:
            unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)

        return unnormed

    def _unnormalize_bounds(self, actions: np.ndarray, stats) -> np.ndarray:
        """Unnormalize using min/max bounds."""
        min_val = getattr(stats, "min", None)
        max_val = getattr(stats, "max", None)
        if min_val is None or max_val is None:
            return actions

        min_val = np.asarray(min_val)
        max_val = np.asarray(max_val)
        dim = min(min_val.shape[-1], actions.shape[-1])

        unnormed = (actions[..., :dim] + 1.0) / 2.0 * (max_val[..., :dim] - min_val[..., :dim] + 1e-8) + min_val[
            ..., :dim
        ]

        if actions.shape[-1] > dim:
            unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)

        return unnormed

    def _unnormalize_normal(self, actions: np.ndarray, stats) -> np.ndarray:
        """Unnormalize using mean/std normalization."""
        mean = getattr(stats, "mean", None)
        std = getattr(stats, "std", None)
        if mean is None or std is None:
            return actions

        mean = np.asarray(mean)
        std = np.asarray(std)
        dim = min(mean.shape[-1], actions.shape[-1])

        unnormed = actions[..., :dim] * (std[..., :dim] + 1e-6) + mean[..., :dim]

        if actions.shape[-1] > dim:
            unnormed = np.concatenate([unnormed, actions[..., dim:]], axis=-1)

        return unnormed


@dataclasses.dataclass(frozen=True)
class StandardOutputDecoder:
    """Default decoder for language-action outputs."""

    def decode(self, *, transform: CoTOutputs, data: dict, reasoning: str) -> dict:
        return transform._process_standard_output(data, reasoning)


@dataclasses.dataclass(frozen=True)
class VLA0OutputDecoder:
    """Decoder for VLA-0 outputs."""

    def decode(self, *, transform: CoTOutputs, data: dict, reasoning: str) -> dict:
        del data
        from lap.policies.lang_action_formats import VLA0ActionFormat

        if isinstance(transform.language_action_format, VLA0ActionFormat):
            actions = transform.language_action_format.parse_to_full_actions(reasoning)
            actions = transform._unnormalize_vla0_actions(actions)
            return {"actions": actions, "reasoning": reasoning}

        movement, gripper_action = transform.language_action_format.parse_language_to_deltas(reasoning)
        if gripper_action is not None:
            single_action = np.concatenate([movement, [gripper_action]])
        else:
            single_action = movement
        return {"actions": single_action, "reasoning": reasoning}