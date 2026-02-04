import re


def is_number(piece: str) -> bool:
    """Check if token contains digits."""
    return bool(re.search(r"[0-9]", piece))


def is_direction_natural(piece: str) -> bool:
    """Check if token contains directional words (for natural language formats)."""
    piece_lower = piece.lower()
    directional_words = ["right", "left", "forward", "up", "down", "back", "clockwise", "counterclockwise"]
    return any(word in piece_lower for word in directional_words)


def is_direction_schema(piece: str) -> bool:
    """Check if token contains +/- symbols (for schema-based formats)."""
    return "+" in piece or "-" in piece


def is_direction_none(piece: str) -> bool:
    """No direction tokens."""
    return False


def is_critical_directional(piece: str) -> bool:
    """Check if token contains digits or directional words (for natural language formats)."""
    return is_number(piece) or is_direction_natural(piece)


def is_critical_schema(piece: str) -> bool:
    """Check if token contains digits or +/- symbols (for schema-based formats)."""
    return is_number(piece) or is_direction_schema(piece)


def is_critical_default(piece: str) -> bool:
    """Default critical token checker - only digits."""
    return is_number(piece)
