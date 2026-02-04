"""Sample-type-specific handlers for CoT policy.

Provides handlers for different sample types:
- VQA samples (image captioning, visual QA)
- Prediction samples (action prediction training with diverse questions)
- Regular robot samples (standard robot action training)
"""

import dataclasses
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from lap.policies.question_types import AnswerFormat
from lap.policies.question_types import QuestionConfig
from lap.policies.question_types import QuestionType
from lap.policies.question_types import compute_dominant_directions
from lap.policies.question_types import compute_gripper_change
from lap.policies.question_types import compute_motion_magnitude
from lap.policies.question_types import format_delta_motion
from lap.policies.question_types import get_embodiment_name
from lap.policies.transforms.action_processor import ActionProcessor
from lap.policies.transforms.action_text import describe_language_action_scale
from lap.policies.transforms.action_text import is_idle_language_action
from lap.policies.transforms.text_utils import TextParser

if TYPE_CHECKING:
    from lap.policies.lang_action_formats import LanguageActionFormat


class RobotSampleProcessingStrategy(Protocol):
    def process(
        self,
        *,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
        handler: "RobotSampleHandler",
    ) -> dict: ...


@dataclasses.dataclass
class VQASampleHandler:
    """Handler for VQA (Visual Question Answering) samples.

    VQA samples include image captioning and visual QA datasets like COCO.
    """

    enable_diverse_questions: bool = False

    def process(self, data: dict, inputs: dict) -> dict:
        """Process a VQA sample.

        Args:
            data: Raw data dictionary.
            inputs: Prepared inputs dictionary.

        Returns:
            Updated inputs dictionary.
        """
        # Extract caption as language action
        caption = TextParser.parse_caption(data)
        inputs["language_actions"] = caption

        # VQA samples are always active (no idle filtering)
        inputs["sample_mask"] = True

        return inputs


@dataclasses.dataclass
class PredictionSampleHandler:
    """Handler for prediction training samples with diverse questions.

    Supports multiple question types for prediction training:
    - Delta motion prediction
    - Task prediction (inverse)
    - Direction classification
    - Gripper prediction
    - Magnitude estimation
    - Temporal ordering
    - Embodiment identification
    """

    question_config: QuestionConfig
    action_processor: ActionProcessor

    def process(
        self,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
    ) -> dict:
        """Process a prediction sample with diverse question generation.

        Args:
            data: Raw data dictionary.
            inputs: Prepared inputs dictionary.
            dataset_name: Name of the source dataset.
            rotation_applied: Whether wrist rotation was applied.

        Returns:
            Updated inputs dictionary.
        """
        raw_lang_actions = data.get("language_actions")
        if raw_lang_actions is None:
            inputs["sample_mask"] = True
            return inputs

        raw_lang_actions = np.asarray(raw_lang_actions, dtype=float)
        initial_state = np.asarray(data.get("raw_state", np.zeros(10)))
        has_wrist_image = data.get("has_wrist_image", False)

        # Transform to appropriate frame
        transformed_actions, frame_description = self.action_processor.transform_to_frame(
            raw_lang_actions, initial_state, dataset_name, rotation_applied, has_wrist_image
        )

        # Extract motion components
        motion_components = ActionProcessor.extract_motion_components(transformed_actions)

        # Sample and format question
        rng = np.random.default_rng()
        question_type = self.question_config.sample_question_type(rng)

        question_prompt, answer = self._format_question_answer(
            data=data,
            inputs=inputs,
            question_type=question_type,
            motion_components=motion_components,
            dataset_name=dataset_name,
            initial_state=initial_state,
            frame_description=frame_description,
            rng=rng,
        )

        # Handle temporal ordering image swap
        if question_type == QuestionType.TEMPORAL_ORDERING:
            self._handle_temporal_swap(inputs)

        # Clean up internal keys
        inputs.pop("_temporal_swap", None)

        # Update inputs
        inputs["prompt"] = question_prompt
        inputs["language_actions"] = answer
        inputs["frame_description"] = frame_description
        inputs["sample_mask"] = True

        return inputs

    def _format_question_answer(
        self,
        data: dict,
        inputs: dict,
        question_type: QuestionType,
        motion_components: dict,
        dataset_name: str,
        initial_state: np.ndarray,
        frame_description: str,
        rng: np.random.Generator,
    ) -> tuple[str, str]:
        """Format question and answer based on question type.

        Args:
            data: Raw data dictionary.
            inputs: Inputs dictionary (may be modified for temporal ordering).
            question_type: Type of question to generate.
            motion_components: Extracted motion components.
            dataset_name: Source dataset name.
            initial_state: Initial robot state.
            frame_description: Frame description string.
            rng: Random number generator.

        Returns:
            Tuple of (question_prompt, answer).
        """
        config = self.question_config

        # Extract motion values
        dx_cm = motion_components["dx_cm"]
        dy_cm = motion_components["dy_cm"]
        dz_cm = motion_components["dz_cm"]
        droll_deg = motion_components["droll_deg"]
        dpitch_deg = motion_components["dpitch_deg"]
        dyaw_deg = motion_components["dyaw_deg"]
        gripper = motion_components["gripper"]
        gripper_action = "open gripper" if gripper >= 0.5 else "close gripper"

        if question_type == QuestionType.DELTA_MOTION:
            return self._format_delta_motion_qa(
                config, rng, frame_description, dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action
            )

        if question_type == QuestionType.TASK_PREDICTION:
            return self._format_task_prediction_qa(
                config, rng, data, dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action
            )

        if question_type == QuestionType.DIRECTION_CLASSIFICATION:
            prompt = config.get_prompt_template(question_type, rng)
            answer = compute_dominant_directions(dx_cm, dy_cm, dz_cm)
            return prompt, answer

        if question_type == QuestionType.GRIPPER_PREDICTION:
            prompt = config.get_prompt_template(question_type, rng)
            initial_gripper = initial_state[6] if len(initial_state) > 6 else 0.5
            answer = compute_gripper_change(initial_gripper, gripper)
            return prompt, answer

        if question_type == QuestionType.MAGNITUDE_ESTIMATION:
            prompt = config.get_prompt_template(question_type, rng)
            answer = compute_motion_magnitude(dx_cm, dy_cm, dz_cm)
            return prompt, answer

        if question_type == QuestionType.TEMPORAL_ORDERING:
            return self._format_temporal_ordering_qa(
                config, rng, inputs, dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action
            )

        if question_type == QuestionType.EMBODIMENT_IDENTIFICATION:
            prompt = config.get_prompt_template(question_type, rng)
            answer = get_embodiment_name(dataset_name)
            return prompt, answer

        # Default to delta motion
        return self._format_delta_motion_qa(
            config, rng, frame_description, dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action
        )

    def _format_delta_motion_qa(
        self,
        config: QuestionConfig,
        rng: np.random.Generator,
        frame_description: str,
        dx_cm: float,
        dy_cm: float,
        dz_cm: float,
        droll_deg: float,
        dpitch_deg: float,
        dyaw_deg: float,
        gripper_action: str,
    ) -> tuple[str, str]:
        """Format delta motion question and answer."""
        answer_format = config.sample_answer_format(rng)
        prompt = config.get_prompt_template(QuestionType.DELTA_MOTION, rng, frame_description=frame_description)
        answer = format_delta_motion(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, answer_format=answer_format
        )
        return prompt, answer

    def _format_task_prediction_qa(
        self,
        config: QuestionConfig,
        rng: np.random.Generator,
        data: dict,
        dx_cm: float,
        dy_cm: float,
        dz_cm: float,
        droll_deg: float,
        dpitch_deg: float,
        dyaw_deg: float,
        gripper_action: str,
    ) -> tuple[str, str]:
        """Format task prediction question and answer."""
        prompt_template = config.get_prompt_template(QuestionType.TASK_PREDICTION, rng)
        action_desc = format_delta_motion(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, answer_format=AnswerFormat.VERBOSE
        )
        question = prompt_template.format(action=action_desc)
        original_prompt = TextParser.parse_prompt(data)
        return question, original_prompt

    def _format_temporal_ordering_qa(
        self,
        config: QuestionConfig,
        rng: np.random.Generator,
        inputs: dict,
        dx_cm: float,
        dy_cm: float,
        dz_cm: float,
        droll_deg: float,
        dpitch_deg: float,
        dyaw_deg: float,
        gripper_action: str,
    ) -> tuple[str, str]:
        """Format temporal ordering question and answer."""
        prompt_template = config.get_prompt_template(QuestionType.TEMPORAL_ORDERING, rng)
        action_desc = format_delta_motion(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, answer_format=AnswerFormat.VERBOSE
        )
        question = prompt_template.format(action=action_desc)

        # Randomly swap images
        swap_images = rng.random() < 0.5
        inputs["_temporal_swap"] = swap_images
        answer = "second" if swap_images else "first"

        return question, answer

    def _handle_temporal_swap(self, inputs: dict) -> None:
        """Handle image swapping for temporal ordering questions."""
        if not inputs.get("_temporal_swap", False):
            return

        if "image" not in inputs:
            return

        images = inputs["image"]
        image_keys = list(images.keys())

        if len(image_keys) < 2:
            return

        # Swap first two images
        key0, key1 = image_keys[0], image_keys[1]
        images[key0], images[key1] = images[key1], images[key0]

        # Also swap masks if present
        if "image_mask" in inputs:
            masks = inputs["image_mask"]
            masks[key0], masks[key1] = masks[key1], masks[key0]


@dataclasses.dataclass
class RobotSampleHandler:
    """Handler for regular robot training samples.

    Processes standard robot action data with language action summaries.
    """

    language_action_format: "LanguageActionFormat"
    action_processor: ActionProcessor
    enable_langact_training: bool = True
    use_rough_scale: bool = False
    enable_diverse_questions: bool = False
    transform_strategy: Literal["standard", "vla0"] = "standard"

    def process(
        self,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
    ) -> dict:
        """Process a regular robot sample.

        Args:
            data: Raw data dictionary.
            inputs: Prepared inputs dictionary.
            dataset_name: Source dataset name.
            rotation_applied: Whether wrist rotation was applied.

        Returns:
            Updated inputs dictionary.
        """
        return self._select_strategy().process(
            data=data,
            inputs=inputs,
            dataset_name=dataset_name,
            rotation_applied=rotation_applied,
            handler=self,
        )

    def _select_strategy(self) -> RobotSampleProcessingStrategy:
        if self.transform_strategy == "vla0":
            return VLA0RobotSampleStrategy()
        return StandardRobotSampleStrategy()

    def _process_language_actions(
        self,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
    ) -> dict:
        """Process standard language actions."""
        initial_state = np.asarray(data["raw_state"])

        # Summarize actions
        lang_actions, frame_desc = self.action_processor.summarize_language_actions(
            data, "language_actions", initial_state, dataset_name, rotation_applied
        )

        inputs["language_actions"] = lang_actions
        inputs["frame_description"] = frame_desc

        # Apply scale description if enabled
        if self.use_rough_scale:
            inputs["language_actions"] = describe_language_action_scale(inputs["language_actions"])

        # Compute sample mask (filter idle/invalid actions)
        inputs["sample_mask"] = self._compute_sample_mask(inputs)

        return inputs

    def _compute_sample_mask(self, inputs: dict) -> bool:
        """Compute whether sample should be included in training."""
        if self.use_rough_scale:
            return True

        lang_actions = inputs["language_actions"]
        sum_decimal = self.language_action_format.get_sum_decimal()
        include_rotation = self.language_action_format.include_rotation

        # Check for idle actions
        is_idle = is_idle_language_action(lang_actions, sum_decimal, include_rotation)
        return not is_idle


@dataclasses.dataclass
class StandardRobotSampleStrategy:
    """Default robot sample processing strategy."""

    def process(
        self,
        *,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
        handler: RobotSampleHandler,
    ) -> dict:
        if "language_actions" in data and handler.enable_langact_training:
            return handler._process_language_actions(data, inputs, dataset_name, rotation_applied)

        inputs["sample_mask"] = True

        return inputs


@dataclasses.dataclass
class VLA0RobotSampleStrategy:
    """Robot sample strategy for the VLA-0 format."""

    def process(
        self,
        *,
        data: dict,
        inputs: dict,
        dataset_name: str,
        rotation_applied: bool,
        handler: RobotSampleHandler,
    ) -> dict:
        del data, dataset_name, rotation_applied  # Unused for this format.
        if "actions" in inputs:
            normalized_actions = inputs["actions"]
            inputs["language_actions"] = handler.language_action_format.summarize_actions(normalized_actions)
            inputs["frame_description"] = "normalized"
        else:
            inputs["language_actions"] = ""
            inputs["frame_description"] = "normalized"

        inputs["sample_mask"] = True
        return inputs
