"""Episode key extractors for bbox annotation lookup.

This module provides functions for extracting unique keys from JSONL
episode data to match with trajectories during dataset iteration.
"""

import re


def droid_key_extractor(episode_data: dict) -> str | None:
    """Extract episode path key from DROID JSONL entry.

    DROID uses file_path in episode_metadata to identify episodes.
    The path is processed to extract the relative episode path.
    
    Args:
        episode_data: Dictionary parsed from JSONL line.
    
    Returns:
        Episode path string or None if not found.
    """
    file_path = episode_data.get("episode_metadata", {}).get("file_path", "")
    if not file_path:
        return None

    # Extract episode path using the same logic as extract_episode_path_from_file_path
    # Remove prefix up to r2d2-data or r2d2-data-full
    rel = re.sub(r"^.*r2d2-data(?:-full)?/", "", file_path)
    # Remove /trajectory... suffix
    episode_path = re.sub(r"/trajectory.*$", "", rel)

    return episode_path if episode_path else None


def oxe_key_extractor(episode_data: dict) -> str | None:
    """Extract episode identifier key from OXE JSONL entry.

    Uses file_path as the unique key since episode_id/episode_index may not be unique
    across combined datasets (e.g., MolmoAct combines household and tabletop which
    may have overlapping episode_index values).

    Args:
        episode_data: Dictionary parsed from JSONL line.
    
    Returns:
        File path string or None if not found.
    """
    episode_metadata = episode_data.get("episode_metadata", {})
    file_path = episode_metadata.get("file_path")
    if file_path:
        return str(file_path)
    return None


def bridge_key_extractor(episode_data: dict) -> str | None:
    """Extract episode identifier key from Bridge JSONL entry.

    Uses both file_path AND episode_id as the unique key because for Bridge dataset,
    one file (e.g., out.npy) can contain multiple episodes. The episode_id identifies
    which episode within that file.

    Key format: "{file_path}::{episode_id}"

    Args:
        episode_data: Dictionary parsed from JSONL line.
    
    Returns:
        Composite key string or None if required fields are not found.
    """
    episode_metadata = episode_data.get("episode_metadata", {})
    file_path = episode_metadata.get("file_path")
    episode_id = episode_metadata.get("episode_id")
    
    if file_path is not None and episode_id is not None:
        return f"{file_path}::{episode_id}"
    return None
