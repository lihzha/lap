import dataclasses
import functools
import logging
import re
from typing import Literal

import numpy as np

from lap.policies.transforms.frame_transforms import transform_actions_from_eef_frame

# PaliGemma tokenizer location (same as used in models/tokenizer.py)
_PALIGEMMA_TOKENIZER_MODEL_PATH = "gs://big_vision/paligemma_tokenizer.model"


@functools.lru_cache(maxsize=1)
def _get_openvla_action_token_pieces(num_bins: int = 256) -> tuple[str, ...]:
    """Return num_bins token pieces from the end of the PaliGemma vocabulary.

    The last tokens in the BPE vocabulary are the least-used ones and are
    conventionally repurposed as action tokens in OpenVLA-style models.
    Result is cached so the tokenizer is loaded at most once per process.
    """
    import sentencepiece

    from lap.shared import download

    path = download.maybe_download(_PALIGEMMA_TOKENIZER_MODEL_PATH, gs={"token": "anon"})
    with path.open("rb") as f:
        spm = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    vocab_size = spm.vocab_size()
    # bin i → token at (vocab_size - num_bins + i)
    return tuple(spm.id_to_piece(vocab_size - num_bins + i) for i in range(num_bins))


@dataclasses.dataclass(frozen=True)
class LanguageActionFormat:
    """Defines how to format language actions.

    This modular structure allows easy extension to support different
    action description formats and styles.
    """

    name: str
    # Style of formatting
    style: Literal["verbose", "compact", "vla0", "openvla"] = "verbose"
    # For verbose style: decimal places for numeric values
    decimal_places: int = 0
    # Whether to include rotation components in descriptions
    include_rotation: bool = False
    # Optional units for translation (cm, m, mm)
    translation_unit: str = "cm"
    # Whether to represent actions in end effector's frame (relative to first timestep)
    use_eef_frame: bool = False

    def get_sum_decimal(self) -> str:
        """Convert to legacy sum_decimal format for backward compatibility."""
        if self.style == "compact":
            return "compact"
        return f"{self.decimal_places}f"

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse language action(s) into translation/rotation deltas and gripper actions."""
        movement = np.zeros(6, dtype=float)  # [dx, dy, dz, droll, dpitch, dyaw]
        gripper_action = None

        if self.style == "compact":
            if self.include_rotation:
                pattern = re.compile(
                    r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+(\d)>"
                )
                match = pattern.search(reasoning)
                if match:
                    groups = match.groups()
                    dx, dy, dz = groups[0], groups[1], groups[2]
                    movement[:3] = np.array([dx, dy, dz], dtype=float) / 100.0
                    if self.include_rotation:
                        droll, dpitch, dyaw = groups[3], groups[4], groups[5]
                        movement[3:6] = np.array([droll, dpitch, dyaw], dtype=float) * np.pi / 180.0

                    gripper_action = float(groups[-1])
        else:
            reasoning = reasoning.replace("slightly", "1.5 cm").replace("moderately", "5 cm").replace("a lot", "10 cm")
            move_pattern = re.compile(
                rf"move\s+(right|left|forward|backward|back|up|down)(?:\s+([\-\d\.]+)\s*{self.translation_unit})?",
                re.IGNORECASE,
            )

            dx_cm = dy_cm = dz_cm = 0.0
            for match in move_pattern.finditer(reasoning):
                direction = match.group(1).lower()
                value = float(match.group(2)) if match.group(2) is not None else 0.0
                if direction == "forward":
                    dx_cm += value
                elif direction in ("backward", "back"):
                    dx_cm -= value
                elif direction == "left":
                    dy_cm += value
                elif direction == "right":
                    dy_cm -= value
                elif direction == "up":
                    dz_cm += value
                elif direction == "down":
                    dz_cm -= value

            movement[:3] = np.array([dx_cm, dy_cm, dz_cm], dtype=float) / 100.0

            if self.include_rotation:
                rotation_pattern = re.compile(
                    r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
                    re.IGNORECASE,
                )
                droll_deg = dpitch_deg = dyaw_deg = 0.0
                for match in rotation_pattern.finditer(reasoning):
                    rotation_type = match.group(1).lower()
                    value = float(match.group(2))

                    if rotation_type == "tilt left":
                        droll_deg += value
                    elif rotation_type == "tilt right":
                        droll_deg -= value
                    elif rotation_type in {"tilt down", "tilt back"}:
                        dpitch_deg += value
                    elif rotation_type in {"tilt up", "tilt forward"}:
                        dpitch_deg -= value
                    elif rotation_type == "rotate counterclockwise":
                        dyaw_deg += value
                    elif rotation_type == "rotate clockwise":
                        dyaw_deg -= value

                movement[3:6] = [
                    droll_deg * np.pi / 180.0,
                    dpitch_deg * np.pi / 180.0,
                    dyaw_deg * np.pi / 180.0,
                ]

            grip_pattern = re.compile(r"set\s+gripper\s+to\s+([\-+]?\d+\.?\d*)", re.IGNORECASE)
            grip_match = grip_pattern.search(reasoning)
            if "open gripper" in reasoning.lower():
                gripper_action = 1.0
            elif "close gripper" in reasoning.lower():
                gripper_action = 0.0
            elif grip_match:
                gripper_action = float(grip_match.group(1))

        if self.use_eef_frame and initial_state is not None:
            movement = transform_actions_from_eef_frame(movement, initial_state)[0]

        return movement, gripper_action


@dataclasses.dataclass(frozen=True)
class VLA0ActionFormat(LanguageActionFormat):
    """VLA-0 style action format: normalized actions as space-separated integers.

    VLA-0 represents actions directly as discretized integers in [0, num_bins] range,
    serialized as space-separated text. This enables using standard language modeling
    loss without any architectural modifications.

    Example output: "523 127 890 512 512 512 500" for a 7D action

    Reference: "VLA-0: Building State-of-the-Art VLAs with Zero Modification"
    """

    name: str = "vla0"
    style: Literal["vla0"] = "vla0"
    # Number of discretization bins (actions scaled from [-1, 1] to [0, num_bins])
    num_bins: int = 1000
    # Number of timesteps to include in output (uses normalized actions field)
    action_horizon: int = 1
    # Action dimension (typically 7 for 6DOF + gripper)
    action_dim: int = 7

    def get_sum_decimal(self) -> str:
        """VLA0 doesn't use the legacy sum_decimal format."""
        return "vla0"

    def summarize_actions(self, actions: np.ndarray) -> str:
        """Convert normalized actions to VLA0 text format.

        Args:
            actions: Normalized actions in [-1, 1] range.
                     Shape: [action_horizon, action_dim] or [action_dim]

        Returns:
            VLA0 format string: "int1 int2 int3 ..." (space-separated integers)
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions[None, :]

        # Clip to valid range and scale from [-1, 1] to [0, num_bins]
        actions = np.clip(actions, -1.0, 1.0)
        discretized = np.round((actions + 1.0) / 2.0 * self.num_bins).astype(int)
        discretized = np.clip(discretized, 0, self.num_bins)

        # Flatten and serialize as space-separated integers (no delimiters per VLA-0 paper)
        flat = discretized.flatten()
        return " ".join(map(str, flat))

    def parse_language_to_deltas(
        self,
        reasoning: str | list[str],
        *,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float | None]:
        """Parse VLA0 text format back to continuous actions.

        Args:
            reasoning: VLA0 format string like "523 127 890 512 512 512 500" (space-separated integers)
            initial_state: Not used for VLA0 (actions are absolute, not relative)

        Returns:
            Tuple of (actions, gripper_action) where actions has shape [action_dim]
            for single-step or flattened for multi-step
        """
        if isinstance(reasoning, list):
            reasoning = " ".join(reasoning)

        # Extract all integers from the string (space-separated, no delimiters)
        try:
            ints = [int(x) for x in reasoning.split()]
        except ValueError:
            return np.zeros(6, dtype=float), None

        if not ints:
            return np.zeros(6, dtype=float), None

        # Convert back to continuous [-1, 1]
        continuous = np.array(ints, dtype=float) / self.num_bins * 2.0 - 1.0

        # Reshape to [action_horizon, action_dim]
        expected_len = self.action_horizon * self.action_dim
        if len(continuous) < expected_len:
            # Pad with zeros if not enough values
            continuous = np.pad(continuous, (0, expected_len - len(continuous)))
        elif len(continuous) > expected_len:
            continuous = continuous[:expected_len]

        actions = continuous.reshape(self.action_horizon, self.action_dim)

        # For compatibility with existing interface, return first timestep's movement and gripper
        # Movement: first 6 dims, Gripper: 7th dim
        movement = actions[0, :6] if actions.shape[1] >= 6 else np.zeros(6)
        gripper_action = float(actions[0, 6]) if actions.shape[1] >= 7 else None

        return movement, gripper_action

    def parse_to_full_actions(self, reasoning: str) -> np.ndarray:
        """Parse VLA0 text format to full action array.

        Args:
            reasoning: VLA0 format string like "523 127 890 512 512 512 500" (space-separated integers)

        Returns:
            Actions array of shape [action_horizon, action_dim] in [-1, 1] range
        """
        if isinstance(reasoning, list):
            reasoning = " ".join(reasoning)

        # Extract integers from <...> format
        match = re.search(r"([\d\s]+)", reasoning)
        if not match:
            logging.info(f"No match found for VLA0 format: {reasoning}")
            return np.zeros((self.action_horizon, self.action_dim), dtype=float)

        try:
            ints = [int(x) for x in reasoning.split()]
        except ValueError:
            logging.info(f"Failed to parse VLA0 format: {reasoning}")
            return np.zeros((self.action_horizon, self.action_dim), dtype=float)

        if not ints:
            logging.info(f"No integers found in VLA0 format: {reasoning}")
            return np.zeros((self.action_horizon, self.action_dim), dtype=float)

        # Convert back to continuous [-1, 1]
        continuous = np.array(ints, dtype=float) / self.num_bins * 2.0 - 1.0

        # Reshape to [action_horizon, action_dim]
        expected_len = self.action_horizon * self.action_dim
        if len(continuous) < expected_len:
            continuous = np.pad(continuous, (0, expected_len - len(continuous)))
        elif len(continuous) > expected_len:
            continuous = continuous[:expected_len]

        return continuous.reshape(self.action_horizon, self.action_dim)


@dataclasses.dataclass(frozen=True)
class OpenVLAActionFormat(LanguageActionFormat):
    """OpenVLA-style action format.

    Like LAP, actions are first summarized with frame transformation.  Unlike
    LAP, the 7-D action is kept as integers rather than converted to natural
    language.  Each dimension is binned into 256 uniform bins covering the
    range [-128, 128] (cm for translation, degrees for rotation).  Gripper
    uses only bin 0 (close) or bin 1 (open).

    The 256 bins are mapped to the 256 least-used tokens in the PaliGemma
    vocabulary (the last entries of the BPE table) so that action tokens do
    not overlap with common language tokens.

    Bin assignment:
        bin_i covers [i - 128, i - 127) cm/degrees
        bin = clip(floor(value + 128), 0, 255)
    """

    name: str = "openvla"
    style: Literal["openvla"] = "openvla"
    num_bins: int = 256

    def get_sum_decimal(self) -> str:
        return "openvla"

    def _value_to_bin(self, value: float) -> int:
        """Map a cm/degree value to a bin index in [0, num_bins - 1]."""
        return int(np.clip(int(np.floor(value + 128)), 0, self.num_bins - 1))

    def _gripper_to_bin(self, gripper: float) -> int:
        """Map gripper state to bin 0 (close) or bin 1 (open)."""
        return 1 if gripper >= 0.5 else 0

    def motion_to_bins(self, motion_components: dict) -> list[int]:
        """Convert motion components (cm/degrees) to bin indices (length 7)."""
        return [
            self._value_to_bin(motion_components["dx_cm"]),
            self._value_to_bin(motion_components["dy_cm"]),
            self._value_to_bin(motion_components["dz_cm"]),
            self._value_to_bin(motion_components["droll_deg"]),
            self._value_to_bin(motion_components["dpitch_deg"]),
            self._value_to_bin(motion_components["dyaw_deg"]),
            self._gripper_to_bin(motion_components["gripper"]),
        ]

    def bins_to_string(self, bins: list[int]) -> str:
        """Convert bin indices to a space-separated string of action token pieces."""
        pieces = _get_openvla_action_token_pieces(self.num_bins)
        return " ".join(pieces[b] for b in bins)

    def summarize_motion_components(self, motion_components: dict) -> str:
        """Full pipeline: motion components → bin indices → token-piece string."""
        return self.bins_to_string(self.motion_to_bins(motion_components))


# Predefined language action formats

VERBOSE_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="verbose_with_rotation",
    style="verbose",
    decimal_places=0,
    include_rotation=True,
)

VERBOSE_EEF_WITH_ROTATION_FORMAT = LanguageActionFormat(
    name="verbose_eef_with_rotation", style="verbose", decimal_places=0, include_rotation=True, use_eef_frame=True
)

# VLA-0 formats: actions as discretized integers
VLA0_CHUNKED_FORMAT = VLA0ActionFormat(
    name="vla0_chunked",
    num_bins=1000,
    action_horizon=10,
    action_dim=7,
)

OPENVLA_FORMAT = OpenVLAActionFormat(
    name="openvla",
    num_bins=256,
)

LANGUAGE_ACTION_FORMAT_REGISTRY = {
    fmt.name: fmt
    for fmt in [
        VERBOSE_WITH_ROTATION_FORMAT,
        VERBOSE_EEF_WITH_ROTATION_FORMAT,
        VLA0_CHUNKED_FORMAT,
        OPENVLA_FORMAT,
    ]
}


def get_language_action_format(name: str) -> LanguageActionFormat:
    """Get a language action format by name."""
    if name not in LANGUAGE_ACTION_FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown language action format: {name}. Available formats: {list(LANGUAGE_ACTION_FORMAT_REGISTRY.keys())}"
        )
    return LANGUAGE_ACTION_FORMAT_REGISTRY[name]
