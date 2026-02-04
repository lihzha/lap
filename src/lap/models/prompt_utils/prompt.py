from collections.abc import Callable
import dataclasses
import random

import numpy as np

import lap.models.prompt_utils.checkers as checkers
from lap.models.prompt_utils.state import StateDiscretizationConfig
from lap.models.prompt_utils.state import StateModule


@dataclasses.dataclass
class PrefixModule:
    """Module for system prompts and schemas that appear before the task.

    Examples:
    - Schema descriptions (e.g., "Schema: <dx dy dz g>; units cm; ...")
    - Coordinate system descriptions
    - Other system-level instructions
    """

    text: str

    def format_prefix(self) -> str:
        """Return the prefix text."""
        return self.text


@dataclasses.dataclass
class TaskModule:
    """Module for formatting task instructions.

    Handles cleaning and formatting of task prompts.
    """

    template: str = "Task: {prompt}, predict the robot's action in the {frame_description}"
    # Whether to include time horizon instruction when provided
    include_time_horizon: bool = False
    # Template for time horizon instruction
    time_horizon_template: str = (
        "predict the robot's action in the future {time_horizon_seconds} seconds in the {frame_description}"
    )

    def format_task(
        self, prompt: str, time_horizon_seconds: float | None = None, frame_description: str = "robot base frame"
    ) -> str:
        """Format task prompt.

        Args:
            prompt: Raw task instruction

        Returns:
            Formatted task string
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ").rstrip(".")
        if self.include_time_horizon:
            assert time_horizon_seconds is not None, "Time horizon must be provided if include_time_horizon is True"
            cleaned_prompt += ", "
            time_horizon_seconds = round(time_horizon_seconds * 2) / 2.0
            cleaned_prompt += self.time_horizon_template.format(time_horizon_seconds=time_horizon_seconds)
        return self.template.format(prompt=cleaned_prompt, frame_description=frame_description)


@dataclasses.dataclass
class ActionModule:
    """Module for action prefix that appears before action output.

    Examples:s
    - "Action: "

    Can optionally include time horizon information.
    """

    prefix: str = "Action: "

    def format_action_prefix(self) -> str:
        """Return the action prefix, optionally including time horizon.

        Args:
            time_horizon_seconds: Optional time horizon in seconds

        Returns:
            Formatted action prefix string
        """

        return self.prefix


@dataclasses.dataclass
class PromptFormat:
    """Defines how to format prompts for tokenization using modular components.

    This allows easy extension to support different prompt formats by composing
    modules in different ways. Supports both training and prediction prompts.
    """

    name: str
    # Optional modules - None means skip that component
    prefix_module: PrefixModule | None = None
    task_module: TaskModule | None = None
    state_module: StateModule | None = None
    action_module: ActionModule | None = None
    # Separator between components (e.g., ", " or "\n")
    separator: str = ""
    # Function to determine if a token piece is critical for this format
    critical_token_checker: Callable[[str], bool] = checkers.is_critical_default
    # Function to determine if a token piece contains direction information
    direction_token_checker: Callable[[str], bool] = checkers.is_direction_none

    @property
    def include_state(self) -> bool:
        """Check if this format includes state."""
        return self.state_module is not None

    def format_prompt(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        state_type: str | None = None,
        time_horizon_seconds: float | None = None,
        frame_description: str = "robot base frame",
        state_dropout: float = 0.0,
    ) -> str:
        """Format the prompt with optional state, state type, and time horizon.

        Args:
            prompt: The task prompt/instruction
            state: Optional state vector to discretize and include
            state_type: Optional state type ("joint_pos", "eef_pose", "none")
            time_horizon_seconds: Optional time horizon in seconds for action prediction

        Returns:
            Formatted prompt string ready for tokenization
        """
        parts = []

        # Add prefix/schema if present
        if self.prefix_module is not None:
            parts.append(self.prefix_module.format_prefix())

        # Add task
        if self.task_module is not None:
            parts.append(
                self.task_module.format_task(
                    prompt=prompt, time_horizon_seconds=time_horizon_seconds, frame_description=frame_description
                )
            )

        # First determine if dropout applies
        add_state = True
        if self.state_module is None or state is None or (state_dropout > 0.0 and random.random() < state_dropout):
            add_state = False

        if add_state:
            state_str = self.state_module.format_state(state=state, state_type=state_type)
            if state_str:  # Only add if non-empty
                parts.append(state_str)

        # Add action prefix (with optional time horizon)
        if self.action_module is not None:
            parts.append(self.action_module.format_action_prefix())

        return self.separator.join(parts)


LAP_PROMPT_FORMAT = PromptFormat(
    name="lap",
    task_module=TaskModule(include_time_horizon=False),
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=checkers.is_critical_directional,
    direction_token_checker=checkers.is_direction_natural,
)


DEFAULT_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="default_prediction",
    state_module=StateModule(
        discretization=StateDiscretizationConfig(bins=256),
        state_prefix_template="State{state_label}: {state}",
        include_state_type=False,
    ),
    task_module=TaskModule(
        # Use {prompt} to allow diverse prompts from CoTInputs
        # Falls back to default if prompt is empty
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    separator="; ",
    action_module=ActionModule(prefix="Answer: "),
    critical_token_checker=checkers.is_critical_schema,
    direction_token_checker=checkers.is_direction_schema,
)


DEFAULT_VQA_PROMPT_FORMAT = PromptFormat(
    name="default_vqa",
    state_module=None,
    task_module=TaskModule(template="Task: {prompt}", include_time_horizon=False),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# VLA-0 Prompt Formats
# Reference: "VLA-0: Building State-of-the-Art VLAs with Zero Modification"

VLA0_CHUNKED_PROMPT_FORMAT = PromptFormat(
    name="vla0_chunked",
    prefix_module=PrefixModule(
        "Analyze the input image and predict robot actions for the next 10 timesteps. "
        "Each action has 7 dimensions. Output a single sequence of 70 integers (0-1000 each), "
        "representing the 10 timesteps sequentially. Provide only space-separated numbers. Nothing else."
    ),
    task_module=TaskModule(template="Task: {prompt}", include_time_horizon=False),
    state_module=None,
    action_module=ActionModule(prefix=""),
    separator="\n",
    critical_token_checker=checkers.is_number,
    direction_token_checker=checkers.is_direction_none,
)

# ---------------------------------------------------------------------------
# Diverse Prediction Prompt Formats for Different Question Types
# ---------------------------------------------------------------------------

# Task Prediction (Inverse): Given action, predict task
TASK_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="task_prediction",
    state_module=None,  # No state needed for task prediction
    task_module=TaskModule(
        template="Task: {prompt}",  # prompt will be formatted externally with action
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Direction Classification: Predict dominant motion direction
DIRECTION_CLASSIFICATION_PROMPT_FORMAT = PromptFormat(
    name="direction_classification",
    state_module=None,
    task_module=TaskModule(
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=checkers.is_direction_natural,
    direction_token_checker=checkers.is_direction_natural,
)

# Gripper Prediction: Predict gripper state change
GRIPPER_PREDICTION_PROMPT_FORMAT = PromptFormat(
    name="gripper_prediction",
    state_module=None,
    task_module=TaskModule(
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Magnitude Estimation: Estimate motion magnitude
MAGNITUDE_ESTIMATION_PROMPT_FORMAT = PromptFormat(
    name="magnitude_estimation",
    state_module=None,
    task_module=TaskModule(
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Temporal Ordering: Determine which image came first
TEMPORAL_ORDERING_PROMPT_FORMAT = PromptFormat(
    name="temporal_ordering",
    state_module=None,
    task_module=TaskModule(
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)

# Embodiment Identification: Identify robot/dataset
EMBODIMENT_IDENTIFICATION_PROMPT_FORMAT = PromptFormat(
    name="embodiment_identification",
    state_module=None,
    task_module=TaskModule(
        template="Task: {prompt}",
        include_time_horizon=False,
    ),
    action_module=ActionModule(prefix="Answer: "),
    separator="; ",
    critical_token_checker=None,
    direction_token_checker=None,
)


# Registry for easy lookup
PROMPT_FORMAT_REGISTRY = {
    "lap": LAP_PROMPT_FORMAT,
    "vla0_chunked": VLA0_CHUNKED_PROMPT_FORMAT,
}

PREDICTION_PROMPT_FORMAT_REGISTRY = {
    "default": DEFAULT_PREDICTION_PROMPT_FORMAT,
    "task_prediction": TASK_PREDICTION_PROMPT_FORMAT,
    "direction_classification": DIRECTION_CLASSIFICATION_PROMPT_FORMAT,
    "gripper_prediction": GRIPPER_PREDICTION_PROMPT_FORMAT,
    "magnitude_estimation": MAGNITUDE_ESTIMATION_PROMPT_FORMAT,
    "temporal_ordering": TEMPORAL_ORDERING_PROMPT_FORMAT,
    "embodiment_identification": EMBODIMENT_IDENTIFICATION_PROMPT_FORMAT,
}
