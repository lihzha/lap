"""Question types and answer format utilities for diverse prediction training.

This module defines:
- QuestionType enum for different QA tasks
- Answer formatters for various output formats
- Prompt variation templates
"""

from dataclasses import dataclass
import enum
import json

import numpy as np


class QuestionType(enum.Enum):
    """Types of questions that can be generated from robot data."""

    # Delta motion prediction (current prediction training)
    DELTA_MOTION = "delta_motion"

    # Inverse: Given action, predict task
    TASK_PREDICTION = "task_prediction"

    # Classify dominant motion direction
    DIRECTION_CLASSIFICATION = "direction_classification"

    # Predict gripper state change
    GRIPPER_PREDICTION = "gripper_prediction"

    # Estimate motion magnitude (qualitative)
    MAGNITUDE_ESTIMATION = "magnitude_estimation"

    # Predict temporal ordering of images
    TEMPORAL_ORDERING = "temporal_ordering"

    # Identify dataset/embodiment from image
    EMBODIMENT_IDENTIFICATION = "embodiment_identification"


class AnswerFormat(enum.Enum):
    """Different formats for answers."""

    VERBOSE = "verbose"  # "move forward 3 cm, move left 2 cm"
    VERBOSE_WITH_ROTATION = "verbose_with_rotation"  # includes rotation
    COMPACT = "compact"  # "<+03 +02 +00 1>"
    COMPACT_WITH_ROTATION = "compact_with_rotation"
    QUALITATIVE = "qualitative"  # "move slightly forward and left"
    COMPONENT = "component"  # "translation: (3, 2, 0) cm; gripper: open"
    JSON = "json"  # {"dx_cm": 3, "dy_cm": 2, ...}
    SENTENCE = "sentence"  # Natural language sentence
    DIRECTION_ONLY = "direction_only"  # "forward, left"


# ---------------------------------------------------------------------------
# Prompt Templates for Different Question Types
# ---------------------------------------------------------------------------

# Delta motion prompts - use {frame_ref} placeholder for frame description
# Frame ref will be filled in based on whether EEF frame transformation is applied
DELTA_MOTION_PROMPTS = [
    "Describe the robot's motion between these two frames{frame_ref}",
    "What movement did the robot make from the first image to the second{frame_ref}?",
    "Predict the change in robot position shown in these images{frame_ref}",
    "Given these before and after images, what action was taken{frame_ref}?",
    "Analyze the visual difference and describe the robot's movement{frame_ref}",
    "What is the delta motion between these two images{frame_ref}?",
    "Describe how the robot end-effector moved between frames{frame_ref}",
    "What movement occurred between these two observations{frame_ref}?",
    "Characterize the robot motion from the image pair{frame_ref}",
    "From image 1 to image 2, describe the robot's action{frame_ref}",
]

TASK_PREDICTION_PROMPTS = [
    "What task is the robot performing given this motion: {action}?",
    "Based on the action '{action}', what is the robot trying to accomplish?",
    "Given the robot moved as follows: {action}, what is the task?",
    "Identify the task from this robot motion: {action}",
    "The robot performed: {action}. What task does this correspond to?",
    "What goal is the robot working towards with this action: {action}?",
]

DIRECTION_CLASSIFICATION_PROMPTS = [
    "What is the dominant motion direction shown in these images?",
    "In which direction(s) did the robot primarily move?",
    "Classify the main movement direction between these frames",
    "What are the primary motion axes in this image pair?",
    "Describe the dominant direction of robot movement",
]

GRIPPER_PREDICTION_PROMPTS = [
    "Did the gripper open, close, or stay the same between these images?",
    "What happened to the gripper state?",
    "Predict the gripper state change from image 1 to image 2",
    "How did the gripper position change?",
    "Was there a gripper action between these frames?",
]

MAGNITUDE_ESTIMATION_PROMPTS = [
    "How much did the robot move between these images?",
    "Estimate the magnitude of the robot's motion",
    "Is the movement between these frames small, moderate, or large?",
    "Characterize the distance traveled by the robot",
    "What is the scale of the robot's displacement?",
]

TEMPORAL_ORDERING_PROMPTS = [
    "Given the robot action '{action}', which image shows the earlier state - the first or second image?",
    "The robot performed: {action}. In what order do these images appear in the trajectory?",
    "Between these frames the robot did: {action}. Which frame came first chronologically?",
    "Given the motion '{action}', determine the temporal order of these two observations",
    "The robot moved as follows: {action}. Is image 1 before or after image 2 in the sequence?",
]

EMBODIMENT_IDENTIFICATION_PROMPTS = [
    "What robot or dataset is this image from?",
    "Identify the robot embodiment shown in this image",
    "What type of robot is performing this task?",
    "Which dataset does this observation come from?",
    "Classify the robot platform shown here",
]


# ---------------------------------------------------------------------------
# Answer Formatting Functions
# ---------------------------------------------------------------------------


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    """Round a value to the nearest multiple of n."""
    return int(round(value / n) * n)


def _format_numeric(val: float, decimals: int = 0) -> str:
    """Format numeric value with specified decimals."""
    return f"{val:.{decimals}f}"


def _num_to_words(n: int) -> str:
    """Convert small integer to word form."""
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
    }
    if n in words:
        return words[n]
    return str(n)


def format_delta_motion_verbose(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
    decimals: int = 0,
) -> str:
    """Format delta motion in verbose style.

    Example: "move forward 3 cm, move left 2 cm, open gripper"
    """
    parts = []

    dx = round(abs(dx_cm), decimals)
    dy = round(abs(dy_cm), decimals)
    dz = round(abs(dz_cm), decimals)

    if dx_cm > 0 and dx != 0:
        parts.append(f"move forward {_format_numeric(dx, decimals)} cm")
    elif dx_cm < 0 and dx != 0:
        parts.append(f"move back {_format_numeric(dx, decimals)} cm")

    if dz_cm > 0 and dz != 0:
        parts.append(f"move up {_format_numeric(dz, decimals)} cm")
    elif dz_cm < 0 and dz != 0:
        parts.append(f"move down {_format_numeric(dz, decimals)} cm")

    if dy_cm > 0 and dy != 0:
        parts.append(f"move left {_format_numeric(dy, decimals)} cm")
    elif dy_cm < 0 and dy != 0:
        parts.append(f"move right {_format_numeric(dy, decimals)} cm")

    if include_rotation:
        droll = _round_to_nearest_n(abs(droll_deg), 10)
        dpitch = _round_to_nearest_n(abs(dpitch_deg), 10)
        dyaw = _round_to_nearest_n(abs(dyaw_deg), 10)

        if droll_deg > 0 and droll != 0:
            parts.append(f"tilt left {droll} degrees")
        elif droll_deg < 0 and droll != 0:
            parts.append(f"tilt right {droll} degrees")

        if dpitch_deg > 0 and dpitch != 0:
            parts.append(f"tilt back {dpitch} degrees")
        elif dpitch_deg < 0 and dpitch != 0:
            parts.append(f"tilt forward {dpitch} degrees")

        if dyaw_deg > 0 and dyaw != 0:
            parts.append(f"rotate counterclockwise {dyaw} degrees")
        elif dyaw_deg < 0 and dyaw != 0:
            parts.append(f"rotate clockwise {dyaw} degrees")

    if gripper_action:
        parts.append(gripper_action)

    return ", ".join(parts) if parts else "no movement"


def format_delta_motion_compact(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_binary: int = 0,
    include_rotation: bool = False,
) -> str:
    """Format delta motion in compact style.

    Example: "<+03 +02 -01 1>" or "<+03 +02 -01 +05 -10 +00 1>"
    """
    dx = int(round(dx_cm))
    dy = int(round(dy_cm))
    dz = int(round(dz_cm))

    parts = [f"{dx:+03d}", f"{dy:+03d}", f"{dz:+03d}"]

    if include_rotation:
        dr = _round_to_nearest_n(droll_deg, 5)
        dp = _round_to_nearest_n(dpitch_deg, 5)
        dy_rot = _round_to_nearest_n(dyaw_deg, 5)
        parts.extend([f"{dr:+03d}", f"{dp:+03d}", f"{dy_rot:+03d}"])

    parts.append(str(gripper_binary))

    return "<" + " ".join(parts) + ">"


def format_delta_motion_qualitative(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
) -> str:
    """Format delta motion in qualitative style.

    Example: "move slightly forward and left, then open gripper"
    """

    def describe_magnitude(cm_value: float) -> str:
        cm_abs = abs(cm_value)
        if cm_abs < 1.5:
            return "slightly"
        if cm_abs < 5:
            return "moderately"
        return "significantly"

    def describe_rotation_magnitude(deg_value: float) -> str:
        deg_abs = abs(deg_value)
        if deg_abs < 10:
            return "slightly"
        if deg_abs < 30:
            return "moderately"
        return "significantly"

    parts = []

    # Translation
    trans_parts = []
    if abs(dx_cm) >= 0.5:
        direction = "forward" if dx_cm > 0 else "backward"
        trans_parts.append(f"{describe_magnitude(dx_cm)} {direction}")

    if abs(dy_cm) >= 0.5:
        direction = "left" if dy_cm > 0 else "right"
        trans_parts.append(f"{describe_magnitude(dy_cm)} {direction}")

    if abs(dz_cm) >= 0.5:
        direction = "up" if dz_cm > 0 else "down"
        trans_parts.append(f"{describe_magnitude(dz_cm)} {direction}")

    if trans_parts:
        parts.append("move " + " and ".join(trans_parts))

    # Rotation (if included)
    if include_rotation:
        rot_parts = []
        if abs(droll_deg) >= 5:
            direction = "tilt left" if droll_deg > 0 else "tilt right"
            rot_parts.append(f"{describe_rotation_magnitude(droll_deg)} {direction}")

        if abs(dpitch_deg) >= 5:
            direction = "tilt back" if dpitch_deg > 0 else "tilt forward"
            rot_parts.append(f"{describe_rotation_magnitude(dpitch_deg)} {direction}")

        if abs(dyaw_deg) >= 5:
            direction = "rotate counterclockwise" if dyaw_deg > 0 else "rotate clockwise"
            rot_parts.append(f"{describe_rotation_magnitude(dyaw_deg)} {direction}")

        if rot_parts:
            parts.append(" and ".join(rot_parts))

    # Gripper
    if gripper_action:
        if parts:
            parts.append(f"then {gripper_action}")
        else:
            parts.append(gripper_action)

    return ", ".join(parts) if parts else "remain stationary"


def format_delta_motion_component(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
    decimals: int = 1,
) -> str:
    """Format delta motion in component style.

    Example: "translation: (3.0, 2.0, 0.0) cm; gripper: open"
    """
    parts = []

    dx = round(dx_cm, decimals)
    dy = round(dy_cm, decimals)
    dz = round(dz_cm, decimals)
    parts.append(f"translation: ({dx}, {dy}, {dz}) cm")

    if include_rotation:
        dr = round(droll_deg, decimals)
        dp = round(dpitch_deg, decimals)
        dyw = round(dyaw_deg, decimals)
        parts.append(f"rotation: ({dr}, {dp}, {dyw}) deg")

    if gripper_action:
        parts.append(f"gripper: {gripper_action}")

    return "; ".join(parts)


def format_delta_motion_json(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
    decimals: int = 1,
) -> str:
    """Format delta motion as JSON.

    Example: {"dx_cm": 3.0, "dy_cm": 2.0, "dz_cm": 0.0, "gripper": "open"}
    """
    data = {
        "dx_cm": round(dx_cm, decimals),
        "dy_cm": round(dy_cm, decimals),
        "dz_cm": round(dz_cm, decimals),
    }

    if include_rotation:
        data["droll_deg"] = round(droll_deg, decimals)
        data["dpitch_deg"] = round(dpitch_deg, decimals)
        data["dyaw_deg"] = round(dyaw_deg, decimals)

    if gripper_action:
        data["gripper"] = gripper_action

    return json.dumps(data)


def format_delta_motion_sentence(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
) -> str:
    """Format delta motion as a natural language sentence.

    Example: "The robot moved forward by three centimeters and left by two centimeters while opening the gripper."
    """
    parts = []

    dx = int(round(abs(dx_cm)))
    dy = int(round(abs(dy_cm)))
    dz = int(round(abs(dz_cm)))

    if dx >= 1:
        direction = "forward" if dx_cm > 0 else "backward"
        parts.append(f"{direction} by {_num_to_words(dx)} centimeter{'s' if dx != 1 else ''}")

    if dy >= 1:
        direction = "left" if dy_cm > 0 else "right"
        parts.append(f"{direction} by {_num_to_words(dy)} centimeter{'s' if dy != 1 else ''}")

    if dz >= 1:
        direction = "up" if dz_cm > 0 else "down"
        parts.append(f"{direction} by {_num_to_words(dz)} centimeter{'s' if dz != 1 else ''}")

    if not parts:
        sentence = "The robot remained stationary"
    elif len(parts) == 1:
        sentence = f"The robot moved {parts[0]}"
    elif len(parts) == 2:
        sentence = f"The robot moved {parts[0]} and {parts[1]}"
    else:
        sentence = f"The robot moved {', '.join(parts[:-1])}, and {parts[-1]}"

    if gripper_action:
        if gripper_action == "open gripper":
            sentence += " while opening the gripper"
        elif gripper_action == "close gripper":
            sentence += " while closing the gripper"

    return sentence + "."


def format_delta_motion_direction_only(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    include_rotation: bool = False,
) -> str:
    """Format delta motion showing only directions (no magnitudes).

    Example: "forward, left, open gripper"
    """
    parts = []

    if abs(dx_cm) >= 0.5:
        parts.append("forward" if dx_cm > 0 else "backward")

    if abs(dy_cm) >= 0.5:
        parts.append("left" if dy_cm > 0 else "right")

    if abs(dz_cm) >= 0.5:
        parts.append("up" if dz_cm > 0 else "down")

    if include_rotation:
        if abs(droll_deg) >= 5:
            parts.append("tilt left" if droll_deg > 0 else "tilt right")
        if abs(dpitch_deg) >= 5:
            parts.append("tilt back" if dpitch_deg > 0 else "tilt forward")
        if abs(dyaw_deg) >= 5:
            parts.append("rotate counterclockwise" if dyaw_deg > 0 else "rotate clockwise")

    if gripper_action:
        parts.append(gripper_action)

    return ", ".join(parts) if parts else "no movement"


# ---------------------------------------------------------------------------
# Answer Generators for Different Question Types
# ---------------------------------------------------------------------------


def compute_dominant_directions(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    threshold_cm: float = 1.0,
) -> str:
    """Compute dominant motion directions."""
    directions = []

    if dx_cm > threshold_cm:
        directions.append("forward")
    elif dx_cm < -threshold_cm:
        directions.append("backward")

    if dy_cm > threshold_cm:
        directions.append("left")
    elif dy_cm < -threshold_cm:
        directions.append("right")

    if dz_cm > threshold_cm:
        directions.append("up")
    elif dz_cm < -threshold_cm:
        directions.append("down")

    if not directions:
        return "stationary"

    return " and ".join(directions)


def compute_gripper_change(gripper_start: float, gripper_end: float) -> str:
    """Compute gripper state change."""
    if gripper_end > 0.5 and gripper_start <= 0.5:
        return "opened"
    if gripper_end <= 0.5 and gripper_start > 0.5:
        return "closed"
    return "unchanged"


def compute_motion_magnitude(dx_cm: float, dy_cm: float, dz_cm: float) -> str:
    """Compute qualitative motion magnitude."""
    l2_norm = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)

    if l2_norm < 2.0:
        return "small movement"
    if l2_norm < 6.0:
        return "moderate movement"
    return "large movement"


def get_embodiment_name(dataset_name: str) -> str:
    """Map dataset name to human-readable embodiment name."""
    mappings = {
        "droid": "DROID (Franka Panda)",
        "bridge": "Bridge (WidowX)",
        "bridge_dataset": "Bridge (WidowX)",
        "fractal": "Fractal (Google Robot)",
        "rt_1_x": "RT-1 (Google Robot)",
        "kuka": "KUKA Robot",
        "fmb": "FMB (Franka Manipulation Benchmark)",
        "taco_play": "TACO Play",
        "jaco_play": "Jaco Play (Kinova Jaco)",
        "berkeley_autolab_ur5": "Berkeley Autolab (UR5)",
        "furniture_bench": "Furniture Bench (Franka)",
        "austin_buds": "Austin BUDS (Franka)",
        "austin_sirius": "Austin Sirius (Franka)",
        "austin_sailor": "Austin Sailor (Franka)",
        "utaustin_mutex": "UT Austin MUTEX (Franka)",
        "viola": "VIOLA (Franka)",
        "cmu_stretch": "CMU Stretch (Hello Robot)",
        "dobbe": "DOBBE (Hello Robot)",
        "iamlab_cmu_pickup_insert": "CMU IAM Lab (Franka)",
    }

    dataset_lower = dataset_name.lower()
    for key, value in mappings.items():
        if key in dataset_lower:
            return value

    return dataset_name


# ---------------------------------------------------------------------------
# Question Type Sampler
# ---------------------------------------------------------------------------


@dataclass
class QuestionConfig:
    """Configuration for question type sampling."""

    # Weights for each question type (will be normalized)
    type_weights: dict[str, float] = None

    # Weights for answer formats in delta motion
    delta_motion_format_weights: dict[str, float] = None

    # Whether to use diverse prompt variations
    use_diverse_prompts: bool = True

    def __post_init__(self):
        if self.type_weights is None:
            self.type_weights = {
                QuestionType.DELTA_MOTION.value: 0.55,
                QuestionType.TASK_PREDICTION.value: 0.15,
                QuestionType.DIRECTION_CLASSIFICATION.value: 0.15,
                QuestionType.GRIPPER_PREDICTION.value: 0.05,
                QuestionType.MAGNITUDE_ESTIMATION.value: 0.05,
                QuestionType.TEMPORAL_ORDERING.value: 0.05,
            }

        if self.delta_motion_format_weights is None:
            self.delta_motion_format_weights = {
                AnswerFormat.VERBOSE.value: 0.35,
                AnswerFormat.VERBOSE_WITH_ROTATION.value: 0.15,
                AnswerFormat.QUALITATIVE.value: 0.2,
                AnswerFormat.COMPACT.value: 0.0,
                AnswerFormat.COMPACT_WITH_ROTATION.value: 0.05,
                AnswerFormat.COMPONENT.value: 0.08,
                AnswerFormat.JSON.value: 0.05,
                AnswerFormat.SENTENCE.value: 0.05,
                AnswerFormat.DIRECTION_ONLY.value: 0.02,
            }

    def sample_question_type(self, rng: np.random.Generator = None) -> QuestionType:
        """Sample a question type based on configured weights."""
        if rng is None:
            rng = np.random.default_rng()

        types = list(self.type_weights.keys())
        weights = np.array([self.type_weights[t] for t in types])
        weights = weights / weights.sum()

        chosen = rng.choice(types, p=weights)
        return QuestionType(chosen)

    def sample_answer_format(self, rng: np.random.Generator = None) -> AnswerFormat:
        """Sample an answer format for delta motion."""
        if rng is None:
            rng = np.random.default_rng()

        formats = list(self.delta_motion_format_weights.keys())
        weights = np.array([self.delta_motion_format_weights[f] for f in formats])
        weights = weights / weights.sum()

        chosen = rng.choice(formats, p=weights)
        return AnswerFormat(chosen)

    def get_prompt_template(
        self,
        question_type: QuestionType,
        rng: np.random.Generator = None,
        frame_description: str = "",
    ) -> str:
        """Get a prompt template for the given question type.

        Args:
            question_type: Type of question to generate prompt for
            rng: Random number generator for sampling
            frame_description: Frame description to include (e.g., "end-effector frame", "robot base frame")
                              Only used for DELTA_MOTION prompts

        Returns:
            Formatted prompt string
        """
        if rng is None:
            rng = np.random.default_rng()

        prompt_lists = {
            QuestionType.DELTA_MOTION: DELTA_MOTION_PROMPTS,
            QuestionType.TASK_PREDICTION: TASK_PREDICTION_PROMPTS,
            QuestionType.DIRECTION_CLASSIFICATION: DIRECTION_CLASSIFICATION_PROMPTS,
            QuestionType.GRIPPER_PREDICTION: GRIPPER_PREDICTION_PROMPTS,
            QuestionType.MAGNITUDE_ESTIMATION: MAGNITUDE_ESTIMATION_PROMPTS,
            QuestionType.TEMPORAL_ORDERING: TEMPORAL_ORDERING_PROMPTS,
            QuestionType.EMBODIMENT_IDENTIFICATION: EMBODIMENT_IDENTIFICATION_PROMPTS,
        }

        prompts = prompt_lists.get(question_type, DELTA_MOTION_PROMPTS)

        if self.use_diverse_prompts:
            template = rng.choice(prompts)
        else:
            template = prompts[0]  # Return first (canonical) prompt

        # Format frame reference for delta motion prompts
        if question_type == QuestionType.DELTA_MOTION and "{frame_ref}" in template:
            if frame_description:
                frame_ref = f" (in {frame_description})"
            else:
                frame_ref = ""
            template = template.format(frame_ref=frame_ref)

        return template


def format_delta_motion(
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    droll_deg: float = 0,
    dpitch_deg: float = 0,
    dyaw_deg: float = 0,
    gripper_action: str = "",
    answer_format: AnswerFormat = AnswerFormat.VERBOSE,
) -> str:
    """Format delta motion using the specified format."""
    include_rotation = answer_format in (
        AnswerFormat.VERBOSE_WITH_ROTATION,
        AnswerFormat.COMPACT_WITH_ROTATION,
    ) or (
        # Include rotation in other formats if there's significant rotation
        (abs(droll_deg) >= 5 or abs(dpitch_deg) >= 5 or abs(dyaw_deg) >= 5)
        and answer_format in (AnswerFormat.COMPONENT, AnswerFormat.JSON, AnswerFormat.QUALITATIVE)
    )

    # Convert gripper_action string for compact format
    gripper_binary = 1 if "open" in gripper_action.lower() else 0

    formatters = {
        AnswerFormat.VERBOSE: lambda: format_delta_motion_verbose(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=False
        ),
        AnswerFormat.VERBOSE_WITH_ROTATION: lambda: format_delta_motion_verbose(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=True
        ),
        AnswerFormat.COMPACT: lambda: format_delta_motion_compact(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_binary, include_rotation=False
        ),
        AnswerFormat.COMPACT_WITH_ROTATION: lambda: format_delta_motion_compact(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_binary, include_rotation=True
        ),
        AnswerFormat.QUALITATIVE: lambda: format_delta_motion_qualitative(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=include_rotation
        ),
        AnswerFormat.COMPONENT: lambda: format_delta_motion_component(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=include_rotation
        ),
        AnswerFormat.JSON: lambda: format_delta_motion_json(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=include_rotation
        ),
        AnswerFormat.SENTENCE: lambda: format_delta_motion_sentence(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=include_rotation
        ),
        AnswerFormat.DIRECTION_ONLY: lambda: format_delta_motion_direction_only(
            dx_cm, dy_cm, dz_cm, droll_deg, dpitch_deg, dyaw_deg, gripper_action, include_rotation=include_rotation
        ),
    }

    formatter = formatters.get(answer_format, formatters[AnswerFormat.VERBOSE])
    return formatter()
