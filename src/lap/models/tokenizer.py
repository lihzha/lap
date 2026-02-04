from abc import ABC
from abc import abstractmethod
import logging
from typing import Literal, TypeAlias

import numpy as np
from openpi.models import tokenizer as _tokenizer
import sentencepiece
from transformers import AutoProcessor

from lap.models.prompt_utils.checkers import is_number
from lap.models.prompt_utils.prompt import DEFAULT_VQA_PROMPT_FORMAT
from lap.models.prompt_utils.prompt import PREDICTION_PROMPT_FORMAT_REGISTRY
from lap.models.prompt_utils.prompt import PROMPT_FORMAT_REGISTRY
from lap.models.prompt_utils.prompt import PromptFormat
import lap.shared.download as download

# SentencePiece model locations
PALIGEMMA_TOKENIZER_MODEL_PATH = "gs://big_vision/paligemma_tokenizer.model"

# Gemma3 special tokens
GEMMA3_BEGIN_IMAGE_TOKEN = 255999
GEMMA3_END_IMAGE_TOKEN = 262144
GEMMA3_IMAGE_TOKEN = 262145  # Placeholder for image embedding
GEMMA3_EOS_TOKEN = 1  # Actual EOS token (was incorrectly 106 which is <start_of_turn>)
GEMMA3_BOS_TOKEN = 2

# Gemma3 IT (instruction-tuned) conversation tokens
# These are the token IDs for the conversation markers
GEMMA3_START_OF_TURN_TOKEN = 106  # <start_of_turn>
GEMMA3_END_OF_TURN_TOKEN = 107  # <end_of_turn>
GEMMA3_USER_TOKEN = 1645  # user
GEMMA3_MODEL_TOKEN = 2516  # model
GEMMA3_NEWLINE_TOKEN = 108  # \n

# Type aliases to reduce repetition
PromptFormatLiteral: TypeAlias = Literal[
    "lap",
    "vla0_chunked",
]
PredictionFormatLiteral: TypeAlias = Literal["default", "grouped"]


def _load_sentencepiece_tokenizer(model_path: str) -> sentencepiece.SentencePieceProcessor:
    """Load a SentencePiece tokenizer from a local/remote path."""
    path = download.maybe_download(model_path, gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def _resolve_prompt_format(prompt_format: str | PromptFormat) -> PromptFormat:
    """Resolve prompt format from string or PromptFormat instance."""
    if isinstance(prompt_format, str):
        if prompt_format not in PROMPT_FORMAT_REGISTRY:
            raise ValueError(
                f"Unknown prompt format: {prompt_format}. Available formats: {list(PROMPT_FORMAT_REGISTRY.keys())}"
            )
        return PROMPT_FORMAT_REGISTRY[prompt_format]
    return prompt_format


def _resolve_prediction_format(prediction_format: str | PromptFormat) -> PromptFormat:
    """Resolve prediction format from string or PromptFormat instance."""
    if isinstance(prediction_format, str):
        if prediction_format not in PREDICTION_PROMPT_FORMAT_REGISTRY:
            raise ValueError(
                f"Unknown prediction format: {prediction_format}. "
                f"Available formats: {list(PREDICTION_PROMPT_FORMAT_REGISTRY.keys())}"
            )
        return PREDICTION_PROMPT_FORMAT_REGISTRY[prediction_format]
    return prediction_format


class BaseTokenizer(ABC):
    """Abstract base class for chain-of-thought tokenizers.

    Provides common functionality for mask creation and reasoning token handling.
    """

    def _init_formats(
        self,
        prompt_format: PromptFormatLiteral | PromptFormat,
        prediction_format: PredictionFormatLiteral | PromptFormat,
        reasoning_mask_prob: float,
    ) -> None:
        """Initialize prompt formats and reasoning mask probability."""
        self.reasoning_mask_prob = reasoning_mask_prob
        logging.info(f"Use reasoning_mask_prob: {self.reasoning_mask_prob}")
        self._prompt_format = _resolve_prompt_format(prompt_format)
        self._prediction_format = _resolve_prediction_format(prediction_format)
        self._vqa_format = DEFAULT_VQA_PROMPT_FORMAT

    def _resolve_format(
        self,
        is_vqa_sample: bool,
        is_prediction_sample: bool,
    ) -> PromptFormat:
        """Resolve which prompt format to use based on sample type."""
        if is_prediction_sample:
            return self._prediction_format
        if is_vqa_sample:
            return self._vqa_format
        return self._prompt_format

    def _create_base_masks(
        self,
        token_count: int,
        reasoning_start: int,
        reasoning_end: int,
        has_reasoning: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Create base attention and reasoning masks.

        Args:
            token_count: Number of tokens before padding
            reasoning_start: Start index of reasoning tokens
            reasoning_end: End index of reasoning tokens
            has_reasoning: Whether reasoning is present

        Returns:
            (attn_mask, reasoning_mask, token_loss_mask)
        """
        attn_mask = np.zeros(self._max_len, dtype=bool)
        token_loss_mask = np.ones(self._max_len, dtype=bool)

        # Mark all non-pad positions as valid for attention
        attn_mask[:token_count] = True

        if not has_reasoning:
            return attn_mask, None, token_loss_mask

        reasoning_mask = np.zeros(self._max_len, dtype=bool)
        start_idx = max(0, min(self._max_len, reasoning_start))
        end_idx = max(0, min(self._max_len, reasoning_end))
        if end_idx > start_idx:
            reasoning_mask[start_idx:end_idx] = True

        return attn_mask, reasoning_mask, token_loss_mask

    def _apply_reasoning_dropout(
        self,
        token_loss_mask: np.ndarray,
        reasoning_mask: np.ndarray,
        is_vqa_sample: bool,
    ) -> np.ndarray:
        """Apply random dropout to reasoning tokens for regularization.

        Args:
            token_loss_mask: Current token loss mask
            reasoning_mask: Mask indicating reasoning token positions
            is_vqa_sample: Whether this is a VQA sample (skips dropout)

        Returns:
            Updated token_loss_mask
        """
        if not 0.0 <= self.reasoning_mask_prob <= 1.0:
            raise ValueError(f"reasoning_mask_prob must be between 0.0 and 1.0, got {self.reasoning_mask_prob}")

        # Skip reasoning_mask_prob for VQA samples - they need stable loss
        if self.reasoning_mask_prob <= 0.0 or is_vqa_sample:
            return token_loss_mask

        reasoning_indices = np.where(reasoning_mask)[0]
        if len(reasoning_indices) == 0:
            return token_loss_mask

        drop_mask = np.random.rand(len(reasoning_indices)) < self.reasoning_mask_prob
        if np.any(drop_mask):
            drop_indices = reasoning_indices[drop_mask]
            token_loss_mask[drop_indices] = False

        return token_loss_mask

    def _build_number_direction_masks(
        self,
        tokens: list[int],
        reasoning_mask: np.ndarray,
        fmt: PromptFormat,
        is_vqa_sample: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build masks for number and direction tokens.

        Args:
            tokens: Token IDs
            reasoning_mask: Mask indicating reasoning token positions
            fmt: Prompt format for direction token checking
            is_vqa_sample: Whether this is a VQA sample

        Returns:
            (number_mask, direction_mask)
        """
        number_mask = np.zeros(self._max_len, dtype=bool)
        direction_mask = np.zeros(self._max_len, dtype=bool)

        if is_vqa_sample:
            return number_mask, direction_mask

        for i in np.where(reasoning_mask)[0]:
            piece = self._get_token_piece(tokens[i])
            if piece:
                if is_number(piece):
                    number_mask[i] = True
                if fmt.direction_token_checker(piece):
                    direction_mask[i] = True

        return number_mask, direction_mask

    @abstractmethod
    def _get_token_piece(self, token_id: int) -> str:
        """Get the string piece for a token ID."""

    @abstractmethod
    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens to string."""

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode string to tokens."""


class PaligemmaTokenizer(_tokenizer.PaligemmaTokenizer, BaseTokenizer):
    def __init__(
        self,
        max_len: int = 48,
        prompt_format: PromptFormatLiteral | PromptFormat = "lap",
        prediction_format: PredictionFormatLiteral | PromptFormat = "default",
        reasoning_mask_prob: float = 0.0,
    ):
        self._tokenizer = _load_sentencepiece_tokenizer(PALIGEMMA_TOKENIZER_MODEL_PATH)
        self._max_len = max_len
        self._init_formats(prompt_format, prediction_format, reasoning_mask_prob)

    def _get_token_piece(self, token_id: int) -> str:
        """Get the string piece for a token ID."""
        return self._tokenizer.id_to_piece(token_id)

    def tokenize(
        self,
        prompt: str,
        reasoning: str | None = None,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        *,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
        frame_description: str = "robot base frame",
        state_dropout: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
        """Tokenize prompt and reasoning for chain-of-thought model.

        Args:
            prompt: Task description
            reasoning: Optional language actions/reasoning
            state: Optional state vector
            state_type: Optional state type descriptor
            is_vqa_sample: Whether this is a VQA sample
            is_prediction_sample: Whether this is a prediction sample
            time_horizon_seconds: Optional time horizon for predictions
            frame_description: Description of coordinate frame
            state_dropout: Probability of dropping state info

        Returns:
            Tuple of (tokens, attn_mask, reasoning_mask, number_mask, direction_mask, token_loss_mask)
        """
        fmt = self._resolve_format(is_vqa_sample, is_prediction_sample)

        formatted_prompt = fmt.format_prompt(
            prompt,
            state,
            state_type,
            time_horizon_seconds=time_horizon_seconds if not is_vqa_sample else None,
            frame_description=frame_description,
            state_dropout=state_dropout,
        )

        # Tokenize
        pad_id = self._tokenizer.pad_id()
        tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        reasoning_start = len(tokens)
        if reasoning is not None:
            clean_reason = reasoning.strip().replace("_", " ").replace("\n", " ")
            tokens += self._tokenizer.encode(clean_reason, add_bos=False, add_eos=True)
        reasoning_end = len(tokens)

        if len(tokens) > self._max_len:
            tokens = tokens[: self._max_len]
            reasoning_end = min(reasoning_end, self._max_len)

        # Create masks using base class helpers
        attn_mask, reasoning_mask, token_loss_mask = self._create_base_masks(
            len(tokens), reasoning_start, reasoning_end, reasoning is not None
        )

        if reasoning is None:
            number_mask = None
            direction_mask = None
        else:
            token_loss_mask = self._apply_reasoning_dropout(token_loss_mask, reasoning_mask, is_vqa_sample)
            number_mask, direction_mask = self._build_number_direction_masks(tokens, reasoning_mask, fmt, is_vqa_sample)

        # Right pad
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = tokens + [pad_id] * pad_count

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            number_mask,
            direction_mask,
            token_loss_mask,
        )

    def decode(self, tokens: np.ndarray) -> str:
        """Decode tokens back to a string, skipping special tokens and placeholders."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        # Filter out tokens that are out of range for this tokenizer
        vocab_size = self._tokenizer.vocab_size()
        filtered_tokens = [t for t in tokens if 0 <= t < vocab_size]

        return self._tokenizer.decode(filtered_tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        """Encode a string to tokens."""
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


class Gemma3Tokenizer(BaseTokenizer):
    """Tokenizer for Gemma3-based models.

    Uses the Gemma3 tokenizer with different special tokens and image placeholders.
    """

    # Default system message for robot context
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful robot assistant."

    def __init__(
        self,
        max_len: int = 800,
        prompt_format: PromptFormatLiteral | PromptFormat = "lap",
        prediction_format: PredictionFormatLiteral | PromptFormat = "default",
        reasoning_mask_prob: float = 0.0,
        tokenizer_name: str = "google/gemma-3-4b-it",
        num_image_tokens: int = 256,  # 4096 patches / 16 (4x4 pooling) = 256
        num_images: int = 2,  # Number of images (base + wrist)
        tokenizer_model_path: str | None = None,  # Device-specific tokenizer model path
    ):
        logging.info(f"Using Gemma3 tokenizer from {tokenizer_name}")
        if tokenizer_model_path is None:
            raise ValueError(
                "Gemma3 tokenizer path is required. Set `gemma3_tokenizer_path` in "
                "src/lap/training/config.py. The Gemma3 tokenizer model is not publicly "
                "available in a default bucket and must be downloaded manually."
            )
        logging.info(f"Loading Gemma3 tokenizer model from configured path: {tokenizer_model_path}")
        self._tokenizer = _load_sentencepiece_tokenizer(tokenizer_model_path)
        self._max_len = max_len
        self._num_image_tokens = num_image_tokens
        self._num_images = num_images

        # Initialize prompt formats using base class helper
        self._init_formats(prompt_format, prediction_format, reasoning_mask_prob)

        # Initialize Gemma3 special tokens
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize Gemma3 special token IDs."""
        self.bos_token_id = GEMMA3_BOS_TOKEN
        self.eos_token_id = GEMMA3_EOS_TOKEN
        self.begin_image_token_id = GEMMA3_BEGIN_IMAGE_TOKEN
        self.end_image_token_id = GEMMA3_END_IMAGE_TOKEN
        self.image_token_id = GEMMA3_IMAGE_TOKEN

        # Conversation tokens for IT format
        self.start_of_turn_token_id = GEMMA3_START_OF_TURN_TOKEN
        self.end_of_turn_token_id = GEMMA3_END_OF_TURN_TOKEN
        self.user_token_id = GEMMA3_USER_TOKEN
        self.model_token_id = GEMMA3_MODEL_TOKEN
        self.newline_token_id = GEMMA3_NEWLINE_TOKEN

    def _get_token_piece(self, token_id: int) -> str:
        """Get the string piece for a token ID."""
        return self._tokenizer.decode([token_id])

    def _build_image_placeholder(self) -> list[int]:
        """Build the image placeholder token sequence for Gemma3.

        For each image: [BEGIN_IMAGE_TOKEN] + [IMAGE_TOKEN] * num_image_tokens + [END_IMAGE_TOKEN]
        """
        single_image_tokens = (
            [self.begin_image_token_id] + [self.image_token_id] * self._num_image_tokens + [self.end_image_token_id]
        )
        return single_image_tokens * self._num_images

    def _build_user_turn_start(self) -> list[int]:
        """Build the start of user turn tokens: <start_of_turn>user\n"""
        return [self.start_of_turn_token_id, self.user_token_id, self.newline_token_id]

    def _build_user_turn_end(self) -> list[int]:
        """Build the end of user turn tokens: <end_of_turn>\n"""
        return [self.end_of_turn_token_id, self.newline_token_id]

    def _build_model_turn_start(self) -> list[int]:
        """Build the start of model turn tokens: <start_of_turn>model\n"""
        return [self.start_of_turn_token_id, self.model_token_id, self.newline_token_id]

    def _build_model_turn_end(self) -> list[int]:
        """Build the end of model turn tokens: <end_of_turn>"""
        return [self.end_of_turn_token_id]

    def _build_user_turn(self, content_tokens: list[int]) -> list[int]:
        """Build a complete user turn with content."""
        return self._build_user_turn_start() + content_tokens + self._build_user_turn_end()

    def _build_gemma3_prefix(self, formatted_prompt: str) -> list[int]:
        """Build the Gemma3 IT format prefix tokens.

        Format: <bos><start_of_turn>user\n[system]\n\n[images]\n[prompt]<end_of_turn>\n<start_of_turn>model\n
        """
        image_tokens = self._build_image_placeholder()
        system_encoded = self._tokenizer.encode(self.DEFAULT_SYSTEM_MESSAGE, add_bos=False, add_eos=False)
        prompt_encoded = self._tokenizer.encode(formatted_prompt, add_bos=False, add_eos=False)

        # Build user turn content
        user_content = (
            system_encoded
            + [self.newline_token_id, self.newline_token_id]
            + image_tokens
            + [self.newline_token_id]
            + prompt_encoded
        )

        return [self.bos_token_id] + self._build_user_turn(user_content) + self._build_model_turn_start()

    def tokenize(
        self,
        prompt: str,
        reasoning: str | None = None,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        *,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
        frame_description: str = "robot base frame",
        state_dropout: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
        """Tokenize prompt and reasoning for Gemma3 chain-of-thought model.

        Uses Gemma3 IT (instruction-tuned) format:
        <bos><start_of_turn>user
        [system_message]
        [images]
        [prompt]<end_of_turn>
        <start_of_turn>model
        [reasoning]<end_of_turn><eos>
        """
        fmt = self._resolve_format(is_vqa_sample, is_prediction_sample)

        formatted_prompt = fmt.format_prompt(
            prompt,
            state,
            state_type,
            time_horizon_seconds=time_horizon_seconds if not is_vqa_sample else None,
            frame_description=frame_description,
            state_dropout=state_dropout,
        )

        # Build prefix using helper
        tokens = self._build_gemma3_prefix(formatted_prompt)
        reasoning_start = len(tokens)

        if reasoning is not None:
            clean_reason = reasoning.strip()
            reasoning_encoded = self._tokenizer.encode(clean_reason, add_bos=False, add_eos=False)
            tokens = tokens + reasoning_encoded + self._build_model_turn_end() + [self.eos_token_id]
        reasoning_end = len(tokens)

        if len(tokens) > self._max_len:
            tokens = tokens[: self._max_len]
            reasoning_end = min(reasoning_end, self._max_len)

        # Create masks using base class helpers
        attn_mask, reasoning_mask, token_loss_mask = self._create_base_masks(
            len(tokens), reasoning_start, reasoning_end, reasoning is not None
        )

        if reasoning is None:
            number_mask = None
            direction_mask = None
        else:
            token_loss_mask = self._apply_reasoning_dropout(token_loss_mask, reasoning_mask, is_vqa_sample)
            number_mask, direction_mask = self._build_number_direction_masks(tokens, reasoning_mask, fmt, is_vqa_sample)

        # Right pad with padding token
        pad_id = self._tokenizer.pad_id() if hasattr(self._tokenizer, "pad_id") else 0
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = tokens + [pad_id] * pad_count

        return (
            np.asarray(tokens, dtype=np.int32),
            attn_mask,
            reasoning_mask,
            number_mask,
            direction_mask,
            token_loss_mask,
        )

    def decode(self, tokens: np.ndarray, skip_special_tokens: bool = True) -> str:
        """Decode tokens back to a string, optionally skipping special tokens and placeholders."""
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        if skip_special_tokens:
            # Filter out image and conversation special tokens
            special_tokens = {
                self.begin_image_token_id,
                self.end_image_token_id,
                self.image_token_id,
                self.start_of_turn_token_id,
                self.end_of_turn_token_id,
                self.user_token_id,
                self.model_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            filtered_tokens = [t for t in tokens if t not in special_tokens]
        else:
            filtered_tokens = tokens

        return self._tokenizer.decode(filtered_tokens).strip()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode a string to tokens."""
        # sentencepiece uses add_bos/add_eos directly
        return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


class FASTTokenizerMixin:
    """Mixin providing common FAST tokenizer functionality.

    This mixin provides:
    - Action token <-> vocabulary token mapping
    - Common tokenize_fast logic for VQA/prediction samples
    - Action extraction from model outputs

    Subclasses must define:
    - _tokenizer: The base text tokenizer
    - _fast_tokenizer: The FAST action tokenizer
    - _fast_skip_tokens: Number of reserved tokens to skip
    - _max_len: Maximum sequence length
    - _prompt_format: Prompt format to use
    - tokenize(): Method to tokenize VQA/prediction samples
    """

    def _init_fast_tokenizer(self, fast_tokenizer_path: str, fast_skip_tokens: int = 128) -> None:
        """Initialize the FAST action tokenizer."""
        self._fast_skip_tokens = fast_skip_tokens
        logging.info(f"Loading FAST tokenizer from: {fast_tokenizer_path}")
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

    def _act_tokens_to_vocab_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        """Map FAST action tokens to vocabulary space.

        Maps FAST tokens to the end of vocab, skipping reserved tokens.
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        vocab_size = self._tokenizer.vocab_size()
        return vocab_size - 1 - self._fast_skip_tokens - tokens

    def _vocab_tokens_to_act_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        """Map vocabulary tokens back to FAST action tokens."""
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        vocab_size = self._tokenizer.vocab_size()
        return vocab_size - 1 - self._fast_skip_tokens - tokens

    def _tokenize_vqa_or_prediction_sample(
        self,
        prompt: str,
        state: np.ndarray,
        language_actions: str | None,
        state_type: str | None,
        is_vqa_sample: bool,
        is_prediction_sample: bool,
        time_horizon_seconds: float | None,
        frame_description: str,
        state_dropout: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Handle VQA and prediction samples using language-based tokenization."""
        tokens, attn_mask, reasoning_mask, _number_mask, _direction_mask, token_loss_mask = self.tokenize(
            prompt=prompt,
            reasoning=language_actions,
            state=state,
            state_type=state_type,
            is_vqa_sample=is_vqa_sample,
            is_prediction_sample=is_prediction_sample,
            time_horizon_seconds=time_horizon_seconds,
            frame_description=frame_description,
            state_dropout=state_dropout,
        )
        # Map outputs to FAST format
        ar_mask = reasoning_mask if reasoning_mask is not None else np.zeros(len(tokens), dtype=bool)
        loss_mask = token_loss_mask if token_loss_mask is not None else np.ones(len(tokens), dtype=bool)
        if reasoning_mask is not None:
            loss_mask = np.logical_and(loss_mask, reasoning_mask)

        return (tokens, attn_mask, ar_mask, loss_mask)

    def _pad_and_convert_to_arrays(
        self,
        tokens: list[int],
        token_mask: list[bool],
        ar_mask: list[bool],
        loss_mask: list[bool],
        pad_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Truncate if needed, pad to max_len, and convert to numpy arrays."""
        if len(tokens) > self._max_len:
            logging.warning(
                f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                "Consider increasing the `max_token_len` in your model config."
            )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        # Right pad to max length
        pad_count = self._max_len - len(tokens)
        if pad_count > 0:
            tokens = tokens + [pad_id] * pad_count
            token_mask = token_mask + [False] * pad_count
            ar_mask = ar_mask + [False] * pad_count
            loss_mask = loss_mask + [False] * pad_count

        return (
            np.asarray(tokens, dtype=np.int32),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )


class Gemma3FASTTokenizer(Gemma3Tokenizer, FASTTokenizerMixin):
    """Tokenizer for Gemma3-based models with FAST action tokens.

    Combines Gemma3 image/text prompt processing with FAST action token handling.
    Uses Gemma3 IT format for prompts and images, but FAST action tokens for robot actions.
    """

    def __init__(
        self,
        fast_tokenizer_path: str,
        max_len: int = 800,
        prompt_format: PromptFormatLiteral | PromptFormat = "lap",
        prediction_format: PredictionFormatLiteral | PromptFormat = "default",
        reasoning_mask_prob: float = 0.0,
        tokenizer_name: str = "google/gemma-3-4b-it",
        num_image_tokens: int = 256,
        num_images: int = 2,
        tokenizer_model_path: str | None = None,
        fast_skip_tokens: int = 128,
    ):
        # Initialize Gemma3 tokenizer (parent class)
        super().__init__(
            max_len=max_len,
            prompt_format=prompt_format,
            prediction_format=prediction_format,
            reasoning_mask_prob=reasoning_mask_prob,
            tokenizer_name=tokenizer_name,
            num_image_tokens=num_image_tokens,
            num_images=num_images,
            tokenizer_model_path=tokenizer_model_path,
        )

        # Initialize FAST tokenizer using mixin
        self._init_fast_tokenizer(fast_tokenizer_path, fast_skip_tokens)

    def tokenize_fast(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None = None,
        language_actions: str | None = None,
        state_type: str | None = None,
        *,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
        state_dropout: float = 0.0,
        clip_action: bool = False,
        frame_description: str = "robot base frame",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize prompt and actions for Gemma3 FAST model.

        For VQA and prediction samples: uses Gemma3 IT format with language reasoning.
        For regular robot samples: uses Gemma3 IT format for prompt/images + FAST action tokens.

        Returns:
            (tokens, token_mask, ar_mask, loss_mask)
        """
        # For VQA and prediction samples, use language-based tokenization
        if is_vqa_sample or is_prediction_sample:
            return self._tokenize_vqa_or_prediction_sample(
                prompt,
                state,
                language_actions,
                state_type,
                is_vqa_sample,
                is_prediction_sample,
                time_horizon_seconds,
                frame_description,
                state_dropout,
            )

        # For regular robot samples, use Gemma3 IT format + FAST action tokens
        formatted_prompt = self._prompt_format.format_prompt(
            prompt,
            state,
            state_type,
            time_horizon_seconds=time_horizon_seconds,
            frame_description=frame_description,
            state_dropout=state_dropout,
        )

        # Build prefix using parent's helper
        prefix_tokens = self._build_gemma3_prefix(formatted_prompt)

        # Append FAST action tokens for robot samples
        if actions is not None:
            if clip_action:
                actions = np.clip(actions, -3.0, 3.0)
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_mapped = self._act_tokens_to_vocab_tokens(action_tokens)
            postfix_tokens = action_tokens_mapped.tolist() + self._build_model_turn_end() + [self.eos_token_id]
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        pad_id = self._tokenizer.pad_id() if hasattr(self._tokenizer, "pad_id") else 0
        return self._pad_and_convert_to_arrays(tokens, token_mask, ar_mask, loss_mask, pad_id)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """Extract actions from decoded tokens."""
        if tokens.ndim > 1:
            tokens = tokens[0]
        tokens = tokens.tolist()

        # Find the start of model turn
        model_turn_start = self._build_model_turn_start()
        start_idx = None
        for i in range(len(tokens) - len(model_turn_start) + 1):
            if tokens[i : i + len(model_turn_start)] == model_turn_start:
                start_idx = i + len(model_turn_start)
                break

        if start_idx is None:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Find the end of model turn
        end_turn_tokens = self._build_model_turn_end()
        end_idx = None
        for i in range(start_idx, len(tokens) - len(end_turn_tokens) + 1):
            if tokens[i : i + len(end_turn_tokens)] == end_turn_tokens:
                end_idx = i
                break

        if end_idx is None:
            end_idx = len(tokens)

        action_token_ids = np.array(tokens[start_idx:end_idx], dtype=np.int32)
        if len(action_token_ids) == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        fast_action_tokens = self._vocab_tokens_to_act_tokens(action_token_ids)
        return self._fast_tokenizer.decode(
            [fast_action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]


class FASTTokenizer(PaligemmaTokenizer, FASTTokenizerMixin):
    """Tokenizer for PaliGemma-based models with FAST action tokens."""

    def __init__(self, fast_tokenizer_path: str, **kwargs):
        super().__init__(**kwargs)
        self._init_fast_tokenizer(fast_tokenizer_path, fast_skip_tokens=128)

    def tokenize_fast(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None = None,
        language_actions: str | None = None,
        state_type: str | None = None,
        *,
        is_vqa_sample: bool = False,
        is_prediction_sample: bool = False,
        time_horizon_seconds: float | None = None,
        state_dropout: float = 0.0,
        clip_action: bool = False,
        frame_description: str = "robot base frame",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize prompt and actions for FAST model.

        For VQA and prediction samples: uses language tokens with proper loss masks.
        For regular robot samples: uses FAST-style action token prediction.

        Returns:
            (tokens, token_mask, ar_mask, loss_mask)
        """
        # For VQA and prediction samples, use language-based tokenization
        if is_vqa_sample or is_prediction_sample:
            return self._tokenize_vqa_or_prediction_sample(
                prompt,
                state,
                language_actions,
                state_type,
                is_vqa_sample,
                is_prediction_sample,
                time_horizon_seconds,
                frame_description,
                state_dropout,
            )

        # For regular robot samples, use FAST action token logic
        formatted_prompt = self._prompt_format.format_prompt(
            prompt,
            state,
            state_type,
            time_horizon_seconds=time_horizon_seconds,
            state_dropout=state_dropout,
            frame_description=frame_description,
        )

        # Tokenize prompt
        pad_id = self._tokenizer.pad_id()
        prefix_tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        # Append action tokens for robot samples
        if actions is not None:
            if clip_action:
                actions = np.clip(actions, -3.0, 3.0)
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_mapped = self._act_tokens_to_vocab_tokens(action_tokens)
            postfix_tokens = action_tokens_mapped.tolist() + self._tokenizer.encode("|", add_eos=True)
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        return self._pad_and_convert_to_arrays(tokens, token_mask, ar_mask, loss_mask, pad_id)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """Extract actions from decoded tokens."""
        if tokens.ndim > 1:
            tokens = tokens[0]
        decoded_tokens = self._tokenizer.decode(tokens.tolist())

        raw_action_tokens = np.array(self._tokenizer.encode(decoded_tokens.split("|")[0].strip()))
        action_tokens = self._act_tokens_to_vocab_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]
