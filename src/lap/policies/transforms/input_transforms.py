"""Input transforms for CoT (Chain-of-Thought) policy training and inference."""

import dataclasses

import numpy as np
from openpi import transforms as upstream_transforms
from openpi.models.model import ModelType

from lap.datasets.utils.helpers import ActionEncoding
from lap.models.model_adapter import IMAGE_KEYS
from lap.models.model_adapter import ExtendedModelType
from lap.policies.lang_action_formats import VERBOSE_EEF_WITH_ROTATION_FORMAT
from lap.policies.lang_action_formats import LanguageActionFormat
from lap.policies.lang_action_formats import get_language_action_format
from lap.policies.question_types import QuestionConfig
from lap.policies.transforms.action_processor import ActionProcessor
from lap.policies.transforms.image_handler import ImageHandler
from lap.policies.transforms.sample_handlers import PredictionSampleHandler
from lap.policies.transforms.sample_handlers import RobotSampleHandler
from lap.policies.transforms.sample_handlers import VQASampleHandler
from lap.policies.transforms.text_utils import TextParser


@dataclasses.dataclass(frozen=True)
class CoTInputs(upstream_transforms.DataTransformFn):
    """Transform raw data samples into model-ready inputs.

    Handles image processing, text parsing, and action summarization for
    training CoT models on robot manipulation data.

    Attributes:
        action_dim: Action dimension of the model (used for padding).
        language_action_format: Format specification for action text summaries.
        wrist_image_dropout_prob: Training-time dropout for wrist images.
        model_type: Type of model being trained.
        action_encoding: Encoding used for robot actions.
        enable_langact_training: Whether to generate language action labels.
        use_rough_scale: Use rough scale descriptions instead of precise values.
        random_base_prob: Probability of using base frame instead of EEF frame.
        random_mask_prob: Probability of masking zero images.
        enable_diverse_questions: Enable diverse question types for prediction.
        question_config: Configuration for diverse question generation.
    """

    # Core configuration
    action_dim: int
    language_action_format: LanguageActionFormat = dataclasses.field(
        default_factory=lambda: VERBOSE_EEF_WITH_ROTATION_FORMAT
    )

    # Training configuration
    wrist_image_dropout_prob: float = 0.0
    model_type: ExtendedModelType = ExtendedModelType.LAP
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    enable_langact_training: bool = True

    # Filtering configuration
    use_rough_scale: bool = False
    transform_strategy: str = "standard"

    # Frame configuration
    random_base_prob: float = 0.0
    random_mask_prob: float = 0.0

    # Diverse question configuration
    enable_diverse_questions: bool = False
    question_config: QuestionConfig | None = None

    def __post_init__(self):
        """Initialize derived components after dataclass creation."""
        self._resolve_language_format()
        self._initialize_question_config()

    def _resolve_language_format(self) -> None:
        """Resolve string schema name to LanguageActionFormat instance."""
        if self.language_action_format is None:
            return
        if isinstance(self.language_action_format, LanguageActionFormat):
            return
        # String needs to be resolved to LanguageActionFormat
        schema = get_language_action_format(self.language_action_format)
        object.__setattr__(self, "language_action_format", schema)

    def _initialize_question_config(self) -> None:
        """Initialize question config if diverse questions are enabled."""
        if self.enable_diverse_questions and self.question_config is None:
            object.__setattr__(self, "question_config", QuestionConfig())

    # -------------------------------------------------------------------------
    # Component Factory Methods
    # -------------------------------------------------------------------------

    def _create_image_handler(self) -> ImageHandler:
        """Create image handler with current configuration."""
        return ImageHandler(
            wrist_image_dropout_prob=self.wrist_image_dropout_prob,
            random_mask_prob=self.random_mask_prob,
        )

    def _create_action_processor(self) -> ActionProcessor:
        """Create action processor with current configuration."""
        return ActionProcessor(
            language_action_format=self.language_action_format,
            random_base_prob=self.random_base_prob,
        )

    def _create_vqa_handler(self) -> VQASampleHandler:
        """Create VQA sample handler."""
        return VQASampleHandler(enable_diverse_questions=self.enable_diverse_questions)

    def _create_prediction_handler(self) -> PredictionSampleHandler:
        """Create prediction sample handler."""
        return PredictionSampleHandler(
            question_config=self.question_config,
            action_processor=self._create_action_processor(),
        )

    def _create_robot_handler(self) -> RobotSampleHandler:
        """Create robot sample handler."""
        return RobotSampleHandler(
            language_action_format=self.language_action_format,
            action_processor=self._create_action_processor(),
            enable_langact_training=self.enable_langact_training,
            use_rough_scale=self.use_rough_scale,
            enable_diverse_questions=self.enable_diverse_questions,
            transform_strategy=self.transform_strategy,
        )

    # -------------------------------------------------------------------------
    # Input Preparation
    # -------------------------------------------------------------------------

    def _prepare_inputs(self, data: dict) -> tuple[dict, bool]:
        """Prepare base inputs from raw data.

        Args:
            data: Raw data dictionary with observation, actions, etc.

        Returns:
            Tuple of (inputs_dict, rotation_applied_flag).
        """
        assert self.model_type in {
            ExtendedModelType.LAP,
            ExtendedModelType.LAP_FAST,
            ModelType.PI0_FAST,
        }
        assert "observation" in data

        # Create handlers
        image_handler = self._create_image_handler()

        # Parse metadata
        dataset_name = TextParser.parse_dataset_name(data)
        is_prediction_sample = data.get("is_prediction_sample", False)
        pred_use_primary = data.get("pred_use_primary", False)
        is_vqa_sample = data.get("is_vqa_sample", False)

        # Process images
        base_image = ImageHandler.parse_base_image(data)
        images, image_masks = image_handler.collect_images(
            data,
            base_image,
            is_prediction_sample=is_prediction_sample,
            pred_use_primary=pred_use_primary,
            is_vqa_sample=is_vqa_sample,
        )

        # Override masks for FAST models
        if self.model_type == ExtendedModelType.LAP_FAST:
            image_masks = [np.True_ for _ in image_masks]

        # Build inputs dictionary
        inputs = {
            "state": data["observation"]["state"],
            "image": dict(zip(IMAGE_KEYS, images, strict=True)),
            "image_mask": dict(zip(IMAGE_KEYS, image_masks, strict=True)),
            "prompt": TextParser.parse_prompt(data),
            "is_prediction_sample": is_prediction_sample,
        }

        if dataset_name:
            inputs["dataset_name"] = dataset_name

        # Pad and store actions if present
        if "actions" in data:
            actions = upstream_transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = np.array(actions)

        # Get rotation flag from dataset-level processing
        rotation_applied = data.get("rotation_applied", False)

        return inputs, rotation_applied

    # -------------------------------------------------------------------------
    # Main Transform
    # -------------------------------------------------------------------------

    def __call__(self, data: dict) -> dict:
        """Transform raw data into model-ready inputs.

        Routes to appropriate handler based on sample type:
        - VQA samples -> VQASampleHandler
        - Prediction samples with diverse questions -> PredictionSampleHandler
        - Regular robot samples -> RobotSampleHandler

        Args:
            data: Raw data dictionary from dataset.

        Returns:
            Transformed inputs dictionary for model.
        """
        # Prepare base inputs
        inputs, rotation_applied = self._prepare_inputs(data)

        # Extract metadata
        dataset_name = TextParser.parse_dataset_name(data)
        is_vqa_sample = data.get("is_vqa_sample", False)
        is_prediction_sample = data.get("is_prediction_sample", False)

        # Store common metadata
        inputs["is_vqa_sample"] = is_vqa_sample
        inputs["time_horizon_seconds"] = data.get("time_horizon_seconds")
        inputs["vqa_dataset_id"] = data.get("vqa_dataset_id", 0)

        # Route to appropriate handler
        if is_vqa_sample:
            return self._create_vqa_handler().process(data, inputs)

        if is_prediction_sample:
            # Set default prediction prompt
            inputs["prompt"] = "predict the robot's action between two images in the prediction"

            # Handle diverse questions for prediction samples
            if self.enable_diverse_questions and self.question_config is not None:
                return self._create_prediction_handler().process(data, inputs, dataset_name, rotation_applied)

        # Validate rotation settings for regular samples
        if self.language_action_format.include_rotation:
            assert self.action_encoding == ActionEncoding.EEF_POS, "Rotation only supported for EEF_POS encoding"

        # Process as regular robot sample
        return self._create_robot_handler().process(data, inputs, dataset_name, rotation_applied)
