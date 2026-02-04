"""Action-to-language formatting and filtering utilities."""

import re

import numpy as np


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _format_numeric(val: float, sum_decimal: str) -> str:
    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        if sum_decimal == "nearest_10":
            return str(int(round(val / 10) * 10))
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m:
            decimals = int(m.group(1))
    return f"{val:.{decimals}f}"


def _summarize_compact_numeric_actions(arr_like, include_rotation: bool = False) -> str:
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    dx_cm = int(round(float(arr[..., 0].sum()) * 100.0))
    dy_cm = int(round(float(arr[..., 1].sum()) * 100.0))
    dz_cm = int(round(float(arr[..., 2].sum()) * 100.0))
    parts = [f"{dx_cm:+03d}", f"{dy_cm:+03d}", f"{dz_cm:+03d}"]

    if include_rotation:
        droll_deg = _round_to_nearest_n(float(arr[..., 3].sum()) * 180.0 / np.pi, 5)
        dpitch_deg = _round_to_nearest_n(float(arr[..., 4].sum()) * 180.0 / np.pi, 5)
        dyaw_deg = _round_to_nearest_n(float(arr[..., 5].sum()) * 180.0 / np.pi, 5)
        parts.extend([f"{droll_deg:+03d}", f"{dpitch_deg:+03d}", f"{dyaw_deg:+03d}"])

    g_last = float(arr[-1, 6])
    parts.append(str(1 if g_last >= 0.5 else 0))
    return "<" + " ".join(parts) + ">"


def summarize_numeric_actions(
    arr_like,
    sum_decimal: str,
    include_rotation: bool = False,
    rotation_precision: int = 10,
) -> str | None:
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    if sum_decimal == "compact":
        return _summarize_compact_numeric_actions(arr, include_rotation)

    if sum_decimal in {"no_number", "nearest_10"}:
        decimals = 0
    else:
        decimals = int(re.fullmatch(r"(\d+)f", sum_decimal).group(1))

    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), rotation_precision)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), rotation_precision)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), rotation_precision)

    parts: list[str] = []
    if sum_decimal == "no_number":
        if dx_m > 0 and dx != 0:
            parts.append("move forward")
        elif dx_m < 0 and dx != 0:
            parts.append("move back")
        if dy_m > 0 and dy != 0:
            parts.append("move left")
        if dy_m < 0 and dy != 0:
            parts.append("move right")
        if dz_m > 0 and dz != 0:
            parts.append("move up")
        elif dz_m < 0 and dz != 0:
            parts.append("move down")
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    g_last = float(arr[-1, 6])
    if g_last >= 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    return ", ".join(parts)


def describe_language_action_scale(language_action: str) -> str | None:
    def _describe_translation(cm_value: float) -> str:
        if cm_value <= 3.0:
            return "slightly"
        if cm_value < 8.0:
            return "moderately"
        return "a lot"

    def _describe_rotation(deg_value: float) -> str:
        if deg_value < 10.0:
            return "slightly"
        if deg_value < 30.0:
            return "moderately"
        return "a lot"

    if language_action is None:
        return None
    if not isinstance(language_action, str) or not language_action.strip():
        return language_action

    translation_pattern = re.compile(r"(move\s+(?:forward|back|left|right|up|down))\s+([+\-]?\d+(?:\.\d+)?)\s*cm")
    rotation_pattern = re.compile(
        r"((?:tilt\s+(?:left|right|back|forward))|(?:rotate\s+(?:clockwise|counterclockwise)))\s+([+\-]?\d+(?:\.\d+)?)\s*degrees"
    )

    def _annotate(text: str, pattern: re.Pattern, descriptor_fn) -> str:
        def _replace(match: re.Match) -> str:
            phrase = match.group(1)
            raw_value = match.group(2)
            try:
                magnitude = float(raw_value)
            except ValueError:
                return match.group(0)
            descriptor = descriptor_fn(magnitude)
            return f"{phrase} {descriptor}"

        return pattern.sub(_replace, text)

    with_translation = _annotate(language_action, translation_pattern, _describe_translation)
    return _annotate(with_translation, rotation_pattern, _describe_rotation)


def summarize_bimanual_numeric_actions(arr_like, sum_decimal: str, include_rotation: bool = False) -> str | None:
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 14:
        return None

    left_actions = arr[..., :7]
    right_actions = arr[..., 7:14]

    if sum_decimal == "compact":
        left_compact = _summarize_compact_numeric_actions(left_actions, include_rotation)
        right_compact = _summarize_compact_numeric_actions(right_actions, include_rotation)
        left_values = left_compact[1:-1]
        right_values = right_compact[1:-1]
        return f"<L {left_values} R {right_values}>"

    left_summary = summarize_numeric_actions(left_actions, sum_decimal, include_rotation)
    right_summary = summarize_numeric_actions(right_actions, sum_decimal, include_rotation)

    if left_summary is None or right_summary is None:
        return None
    return f"Left arm: {left_summary}. Right arm: {right_summary}"


def is_idle_language_action(
    language_action: str,
    sum_decimal: str,
    include_rotation: bool = False,
    translation_threshold: float = 1.0,
    rotation_threshold_deg: float = 10.0,
) -> bool:
    if not language_action or not isinstance(language_action, str):
        return True

    if sum_decimal == "compact":
        if include_rotation:
            match = re.search(
                r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>",
                language_action,
            )
            if match:
                dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))
                droll_deg, dpitch_deg, dyaw_deg = int(match.group(4)), int(match.group(5)), int(match.group(6))
                translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
                rotation_l2 = np.sqrt(droll_deg**2 + dpitch_deg**2 + dyaw_deg**2)
                return translation_l2 < translation_threshold and rotation_l2 < rotation_threshold_deg
            return True
        match = re.search(r"<([+\-]\d+)\s+([+\-]\d+)\s+([+\-]\d+)\s+\d>", language_action)
        if match:
            dx_cm, dy_cm, dz_cm = int(match.group(1)), int(match.group(2)), int(match.group(3))
            translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
            return translation_l2 < translation_threshold
        return True

    if sum_decimal == "no_number":
        move_pattern_no_number = re.compile(
            r"move\s+(right|left|forward|backward|back|up|down)(?!\s+[\d.])", re.IGNORECASE
        )
        has_movement = bool(move_pattern_no_number.search(language_action))
        if not include_rotation:
            return not has_movement
        rotation_pattern_no_number = re.compile(
            r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)(?!\s+[\d.])",
            re.IGNORECASE,
        )
        has_rotation = bool(rotation_pattern_no_number.search(language_action))
        return not (has_movement or has_rotation)

    move_pattern = re.compile(r"move\s+(right|left|forward|backward|back|up|down)\s+([\d.]+)\s*cm", re.IGNORECASE)
    dx_cm = dy_cm = dz_cm = 0.0
    for match in move_pattern.finditer(language_action):
        direction = match.group(1).lower()
        value = float(match.group(2))
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

    translation_l2 = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
    if not include_rotation:
        return translation_l2 < translation_threshold

    rotation_pattern = re.compile(
        r"(tilt left|tilt right|tilt up|tilt down|tilt back|tilt forward|rotate clockwise|rotate counterclockwise)\s+([\d.]+)\s*degrees",
        re.IGNORECASE,
    )
    droll_deg = dpitch_deg = dyaw_deg = 0.0
    for match in rotation_pattern.finditer(language_action):
        rotation_type = match.group(1).lower()
        value = float(match.group(2))
        if rotation_type == "tilt left":
            droll_deg += value
        elif rotation_type == "tilt right":
            droll_deg -= value
        elif rotation_type in {"tilt up", "tilt forward"}:
            dpitch_deg += value
        elif rotation_type in {"tilt down", "tilt back"}:
            dpitch_deg -= value
        elif rotation_type == "rotate counterclockwise":
            dyaw_deg += value
        elif rotation_type == "rotate clockwise":
            dyaw_deg -= value

    rotation_l2 = np.sqrt(droll_deg**2 + dpitch_deg**2 + dyaw_deg**2)
    return translation_l2 < translation_threshold and rotation_l2 < rotation_threshold_deg

