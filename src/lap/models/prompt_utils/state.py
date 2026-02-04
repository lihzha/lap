import dataclasses

import numpy as np


@dataclasses.dataclass
class StateTemplate:
    """Template for formatting discretized state values.

    Allows flexible representation of state vectors with custom labels and formatting.
    """

    # Dimension labels in order (e.g., ["x", "y", "z", "rot1x", ...])
    # If None or shorter than values, uses generic labels
    dim_labels: list[str] | None = None

    # Format string for each dimension value
    # Can use {label} and {value} placeholders
    # e.g., "{label}={value:03d}" for "x=134" or just "{value}" for "134"
    dim_format: str = "{value}"

    # Separator between dimensions
    separator: str = " "

    def format_state(self, values: np.ndarray) -> str:
        """Format discretized state values according to template.

        Args:
            values: Array of discretized state values

        Returns:
            Formatted string representation
        """
        parts = []
        for i, val in enumerate(values):
            # Use provided label or generate generic one
            if self.dim_labels and i < len(self.dim_labels):
                label = self.dim_labels[i]
            else:
                label = f"dim{i}"

            parts.append(self.dim_format.format(label=label, value=int(val)))

        return self.separator.join(parts)


@dataclasses.dataclass
class GroupedStateTemplate:
    """Template for formatting state values grouped by semantic meaning.

    Groups dimensions into semantic categories (e.g., position, rotation, gripper).
    Example output: "position 134 088 076, rotation 201 054 233 128 254 033, gripper 200"
    """

    # Group labels (e.g., ["position", "rotation", "gripper"])
    group_labels: list[str]

    # Number of dimensions in each group (e.g., [3, 6, 1] for 10D state)
    group_sizes: list[int]

    # Format string for each value within a group
    # e.g., "{value:03d}" for zero-padded or "{value}" for no padding
    value_format: str = "{value:03d}"

    # Separator between groups
    group_separator: str = ", "

    # Separator between values within a group
    value_separator: str = " "

    def format_state(self, values: np.ndarray) -> str:
        """Format discretized state values with semantic grouping.

        Args:
            values: Array of discretized state values

        Returns:
            Formatted string representation with grouped dimensions
        """
        if sum(self.group_sizes) > len(values):
            # Handle case where state is shorter than expected
            # Adjust group sizes to fit available values
            adjusted_sizes = []
            remaining = len(values)
            for size in self.group_sizes:
                adjusted_sizes.append(min(size, remaining))
                remaining -= adjusted_sizes[-1]
                if remaining <= 0:
                    break
            group_sizes = adjusted_sizes
        else:
            group_sizes = self.group_sizes

        parts = []
        idx = 0
        for label, size in zip(self.group_labels, group_sizes):
            if idx >= len(values):
                break

            # Extract values for this group
            group_values = values[idx : idx + size]
            formatted_values = self.value_separator.join(self.value_format.format(value=int(v)) for v in group_values)

            # Format as "label value1 value2 ..."
            parts.append(f"{label} {formatted_values}")
            idx += size

        return self.group_separator.join(parts)


@dataclasses.dataclass
class StateDiscretizationConfig:
    """Configuration for discretizing state vectors into text."""

    bins: int = 256
    min_dim: int = 10  # Minimum number of dimensions to include (avoid over-trimming)
    range_min: float = -1.0
    range_max: float = 1.0
    template: StateTemplate | GroupedStateTemplate | None = None  # If None, uses default space-separated format

    def discretize_state(self, state: np.ndarray) -> str:
        """Discretize state vector into string representation.

        Trims trailing zero-padded dimensions and discretizes to bins.
        Uses the configured StateTemplate if provided, otherwise defaults to space-separated values.

        Args:
            state: State vector to discretize

        Returns:
            Formatted string representation of discretized state
        """
        state_arr = np.asarray(state)
        eps = 1e-8

        # Trim zero-padded dimensions
        if state_arr.ndim == 1:
            non_zero_mask = np.abs(state_arr) > eps
            last_idx = int(np.nonzero(non_zero_mask)[0][-1]) + 1 if np.any(non_zero_mask) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[:last_idx]
        else:
            flat = state_arr.reshape(-1, state_arr.shape[-1])
            non_zero_cols = np.any(np.abs(flat) > eps, axis=0)
            last_idx = int(np.nonzero(non_zero_cols)[0][-1]) + 1 if np.any(non_zero_cols) else 0
            last_idx = max(last_idx, self.min_dim)
            trimmed = state_arr[..., :last_idx].reshape(-1)

        if trimmed.size > 0:
            bins = np.linspace(self.range_min, self.range_max, self.bins + 1)[:-1]
            discretized_state = np.digitize(trimmed, bins=bins) - 1

            # Use template if provided, otherwise default to space-separated
            if self.template is not None:
                return self.template.format_state(discretized_state)
            return " ".join(map(str, discretized_state))
        return ""


@dataclasses.dataclass
class StateModule:
    """Unified module for state discretization and formatting.

    Handles both training prompts (with state type labels) and prediction prompts.
    Combines discretization configuration with template formatting.
    """

    # Discretization configuration
    discretization: StateDiscretizationConfig

    # Template for state prefix in training/normal prompts
    # Can use {state} and {state_label} placeholders
    state_prefix_template: str = "State{state_label}: {state}"

    # Whether to include state type label in state prefix
    include_state_type: bool = True

    def format_state(
        self,
        state: np.ndarray | None = None,
        state_type: str | None = None,
    ) -> str:
        """Format state for inclusion in prompt.

        Args:
            state: Optional state vector to discretize and format
            state_type: Optional state type ("joint_pos", "eef_pose", "none")
            mode: Whether this is for training or prediction

        Returns:
            Formatted state string ready for inclusion in prompt
        """
        # Handle no state case
        if state is None or state_type == "none":
            if self.include_state_type:
                return self.state_prefix_template.format(state="", state_label="None")
            return self.state_prefix_template.format(state="", state_label="")

        # Discretize state
        state_str = self.discretization.discretize_state(state)

        # Map state_type to human-readable label
        state_type_labels = {
            "joint_pos": " (joint position)",
            "eef_pose": " (end-effector pose)",
        }
        state_label = state_type_labels.get(state_type, state_type) if state_type else ""

        if self.include_state_type:
            return self.state_prefix_template.format(state=state_str, state_label=state_label)
        return self.state_prefix_template.format(state=state_str, state_label="")


# Predefined state templates - easily extensible by adding new instances
DEFAULT_STATE_TEMPLATE = StateTemplate(
    dim_labels=None,
    dim_format="{value}",
    separator=" ",
)

NAMED_PARAMS_STATE_TEMPLATE = StateTemplate(
    dim_labels=["x", "y", "z", "rot1x", "rot1y", "rot1z", "rot2x", "rot2y", "rot2z", "grip"],
    dim_format="{label}={value:03d}",
    separator=" ",
)

VERBOSE_STATE_TEMPLATE = StateTemplate(
    dim_labels=[
        "position_x",
        "position_y",
        "position_z",
        "rotation_1_x",
        "rotation_1_y",
        "rotation_1_z",
        "rotation_2_x",
        "rotation_2_y",
        "rotation_2_z",
        "gripper",
    ],
    dim_format="{label}={value:03d}",
    separator=", ",
)

GROUPED_STATE_TEMPLATE = GroupedStateTemplate(
    group_labels=["position", "rotation", "gripper"],
    group_sizes=[3, 3, 1],
    value_format="{value:03d}",
    group_separator=", ",
    value_separator=" ",
)
