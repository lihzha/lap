"""Action processing utilities for CoT policy."""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

import numpy as np

from lap.policies.transforms.action_text import summarize_bimanual_numeric_actions
from lap.policies.transforms.action_text import summarize_numeric_actions
from lap.policies.transforms.frame_transforms import transform_actions_to_eef_frame

if TYPE_CHECKING:
    from lap.policies.lang_action_formats import LanguageActionFormat


@dataclasses.dataclass
class ActionProcessor:
    """Processes and summarizes robot actions for language action format.

    Attributes:
        language_action_format: Format specification for action summarization.
        random_base_prob: Probability of using base frame instead of EEF frame.
    """

    language_action_format: "LanguageActionFormat"
    random_base_prob: float = 0.0

    def summarize_language_actions(
        self,
        data: dict,
        lang_action_key: str = "language_actions",
        initial_state: np.ndarray | None = None,
        dataset_name: str | None = None,
        rotation_applied: bool = False,
    ) -> tuple[str | None, str]:
        """Summarize raw language actions into text description.

        Args:
            data: Data dictionary containing language actions and metadata.
            lang_action_key: Key to access language actions in data.
            initial_state: Initial robot state for EEF frame transformation.
            dataset_name: Name of the dataset (for dataset-specific handling).
            rotation_applied: Whether wrist rotation was applied at dataset level.

        Returns:
            Tuple of (summarized_text, frame_description).
        """
        language_actions = data[lang_action_key]
        is_bimanual: bool = data.get("is_bimanual", False)
        is_navigation: bool = data.get("is_navigation", False)
        has_wrist_image: bool = data.get("has_wrist_image", False)

        # Determine frame to use
        use_eef_frame, frame_description = self._should_use_eef_frame(initial_state, has_wrist_image)

        # Transform to EEF frame if requested
        if use_eef_frame:
            language_actions = transform_actions_to_eef_frame(
                language_actions, initial_state, dataset_name, rotation_applied
            )

        # Summarize based on robot type
        summed = self._summarize_for_robot_type(
            language_actions,
            is_bimanual=is_bimanual,
            is_navigation=is_navigation,
        )

        return summed, frame_description

    def _should_use_eef_frame(
        self,
        initial_state: np.ndarray | None,
        has_wrist_image: bool,
    ) -> tuple[bool, str]:
        """Determine whether to use EEF frame transformation.

        Args:
            initial_state: Initial robot state.
            has_wrist_image: Whether the sample has a wrist image.

        Returns:
            Tuple of (use_eef_frame, frame_description).
        """
        use_eef_frame = self.language_action_format.use_eef_frame and initial_state is not None

        # Apply random base frame selection during training
        if self.random_base_prob > 0.0:
            use_eef_frame = use_eef_frame and has_wrist_image and random.random() < (1 - self.random_base_prob)

        frame_description = "end-effector frame" if use_eef_frame else "robot base frame"
        return use_eef_frame, frame_description

    def _summarize_for_robot_type(
        self,
        language_actions: np.ndarray,
        *,
        is_bimanual: bool,
        is_navigation: bool,
    ) -> str:
        """Summarize actions based on robot type.

        Args:
            language_actions: Transformed language actions.
            is_bimanual: Whether this is a bimanual robot.
            is_navigation: Whether this is a navigation robot.

        Returns:
            Summarized action text.
        """
        if is_bimanual:
            return summarize_bimanual_numeric_actions(
                language_actions,
                self.language_action_format.get_sum_decimal(),
                self.language_action_format.include_rotation,
            )

        if is_navigation:
            return summarize_numeric_actions(
                language_actions,
                "nearest_10",
                include_rotation=True,
                rotation_precision=10,
            )

        return summarize_numeric_actions(
            language_actions,
            sum_decimal=self.language_action_format.get_sum_decimal(),
            include_rotation=self.language_action_format.include_rotation,
        )

    @staticmethod
    def extract_motion_components(language_actions: np.ndarray) -> dict:
        """Extract motion components from raw language actions.

        Converts raw actions [dx, dy, dz, droll, dpitch, dyaw, gripper] from
        meters/radians to human-readable units (cm/degrees).

        Args:
            language_actions: Raw action array [7] in meters/radians.

        Returns:
            Dictionary with motion components in human-readable units:
            - dx_cm, dy_cm, dz_cm: Translation in cm
            - droll_deg, dpitch_deg, dyaw_deg: Rotation in degrees
            - gripper: Gripper state (0-1)
        """
        arr = np.asarray(language_actions, dtype=float)
        if arr.ndim == 2:
            arr = arr[0]  # Take first timestep if batched

        # Convert to cm and degrees
        motion = {
            "dx_cm": arr[0] * 100.0,  # meters to cm
            "dy_cm": arr[1] * 100.0,
            "dz_cm": arr[2] * 100.0,
            "droll_deg": arr[3] * 180.0 / np.pi if len(arr) > 3 else 0.0,
            "dpitch_deg": arr[4] * 180.0 / np.pi if len(arr) > 4 else 0.0,
            "dyaw_deg": arr[5] * 180.0 / np.pi if len(arr) > 5 else 0.0,
            "gripper": arr[6] if len(arr) > 6 else 0.5,
        }

        return motion

    def transform_to_frame(
        self,
        raw_actions: np.ndarray,
        initial_state: np.ndarray | None,
        dataset_name: str,
        rotation_applied: bool,
        has_wrist_image: bool,
    ) -> tuple[np.ndarray, str]:
        """Transform actions to appropriate frame (EEF or base).

        Args:
            raw_actions: Raw action array.
            initial_state: Initial robot state.
            dataset_name: Dataset name.
            rotation_applied: Whether wrist rotation was applied.
            has_wrist_image: Whether sample has wrist image.

        Returns:
            Tuple of (transformed_actions, frame_description).
        """
        use_eef_frame, frame_description = self._should_use_eef_frame(initial_state, has_wrist_image)

        if use_eef_frame:
            raw_actions = transform_actions_to_eef_frame(raw_actions, initial_state, dataset_name, rotation_applied)

        return raw_actions, frame_description
