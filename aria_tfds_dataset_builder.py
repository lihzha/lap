from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aria_dataset.conversion_utils import MultiThreadedDatasetBuilder
import cv2
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
import zarr


def decode_jpeg(data):
    """Decode JPEG data from zarr to numpy array."""
    # Unwrap nested numpy arrays
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # Convert to bytes if needed
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
        arr = data
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    # Decode JPEG
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    # Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_task_boundaries_from_timestamps(
    timestamps: np.ndarray,
    fps: int = 30,
    gap_threshold: float = 2.0,
    min_task_duration: float = 2.0,
) -> list:
    """
    Detect task instance boundaries from timestamp discontinuities.

    Args:
        timestamps: Array of timestamps in nanoseconds
        fps: Frames per second
        gap_threshold: Minimum gap (seconds) to consider a task boundary
        min_task_duration: Minimum task duration (seconds)

    Returns:
        List of (start_frame, end_frame) tuples for each task instance
    """
    # Compute time deltas in seconds
    time_deltas = np.diff(timestamps) / 1e9

    # Find large gaps
    gap_indices = np.where(time_deltas > gap_threshold)[0]

    # Create task segments
    min_frames = int(min_task_duration * fps)
    tasks = []

    if len(gap_indices) == 0:
        # No gaps - entire episode is one task
        tasks.append((0, len(timestamps) - 1))
    else:
        # First task
        if gap_indices[0] >= min_frames:
            tasks.append((0, gap_indices[0]))

        # Middle tasks
        for i in range(len(gap_indices) - 1):
            start = gap_indices[i] + 1
            end = gap_indices[i + 1]
            if end - start >= min_frames:
                tasks.append((start, end))

        # Last task
        start = gap_indices[-1] + 1
        if len(timestamps) - start >= min_frames:
            tasks.append((start, len(timestamps) - 1))

    return tasks


def _generate_examples(paths) -> Iterator[tuple[str, Any]]:
    """Yields episodes for list of zarr directory paths."""

    # Debug: log what we received
    if not isinstance(paths, list):
        print(f"ERROR: Expected list of paths, got {type(paths)}: {paths}")
        return

    # Filter out any invalid paths
    valid_paths = [p for p in paths if p and isinstance(p, str) and len(p) > 0]
    if len(valid_paths) != len(paths):
        print(f"Warning: Filtered {len(paths) - len(valid_paths)} invalid paths from batch")

    def _parse_example(zarr_path):
        """Parse episodes from a single zarr directory.

        zarr_path format: /path/to/data/aria_fold_clothes/YYYY-MM-DD-HH-MM-SS-XXXXXX
        """
        # Validate path
        if not zarr_path or not isinstance(zarr_path, (str, Path)):
            print(f"Warning: Invalid path type: {type(zarr_path)}, value: {zarr_path}")
            return

        zarr_path = Path(zarr_path)

        # Check if path exists and is a directory
        if not zarr_path.exists():
            print(f"Warning: Path does not exist: {zarr_path}")
            return

        if not zarr_path.is_dir():
            print(f"Warning: Path is not a directory: {zarr_path}")
            return

        # Load zarr store with retry (zarr v3 can have multiprocessing issues)
        import time

        max_retries = 3
        store = None
        for attempt in range(max_retries):
            try:
                # Use zarr.open_group with explicit string path for better zarr v3 compatibility
                store = zarr.open_group(str(zarr_path), mode="r")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"Warning: Failed to open zarr at {zarr_path} after {max_retries} attempts: {e}")
                    return

        if store is None:
            return

        # Load metadata
        try:
            metadata = store.attrs.asdict()
            task_name = metadata.get("task_name", "fold_clothes")
            fps = metadata.get("fps", 30)
        except Exception as e:
            print(f"Warning: Failed to load metadata from {zarr_path}: {e}")
            return

        # Load all observations
        try:
            images = store["images.front_1"][:]
            left_ee_pose = store["left.obs_ee_pose"][:]
            right_ee_pose = store["right.obs_ee_pose"][:]
            left_keypoints = store["left.obs_keypoints"][:]
            right_keypoints = store["right.obs_keypoints"][:]
            left_wrist_pose = store["left.obs_wrist_pose"][:]
            right_wrist_pose = store["right.obs_wrist_pose"][:]
            head_pose = store["obs_head_pose"][:]
            timestamps = store["obs_rgb_timestamps_ns"][:]

            # Optionally load eye gaze if available
            # eye_gaze = None
            # if 'obs_eye_gaze' in store:
            #     eye_gaze = store['obs_eye_gaze'][:]
        except Exception as e:
            print(f"Warning: Failed to load data arrays from {zarr_path}: {e}")
            return

        # Detect task boundaries from timestamps
        task_segments = detect_task_boundaries_from_timestamps(
            timestamps,
            fps=fps,
            gap_threshold=2.0,
            min_task_duration=2.0,
        )

        # Check if images are encoded
        is_encoded = isinstance(images[0], (bytes, np.void)) or (
            isinstance(images[0], np.ndarray) and images[0].dtype == np.uint8 and images[0].ndim == 1
        )

        # Process each task segment
        for seg_idx, (start_frame, end_frame) in enumerate(task_segments):
            num_frames = end_frame - start_frame + 1

            # Extract segment data
            seg_images = images[start_frame : end_frame + 1]
            seg_left_ee = left_ee_pose[start_frame : end_frame + 1]
            seg_right_ee = right_ee_pose[start_frame : end_frame + 1]
            seg_left_kp = left_keypoints[start_frame : end_frame + 1]
            seg_right_kp = right_keypoints[start_frame : end_frame + 1]
            seg_left_wrist = left_wrist_pose[start_frame : end_frame + 1]
            seg_right_wrist = right_wrist_pose[start_frame : end_frame + 1]
            seg_head = head_pose[start_frame : end_frame + 1]

            # seg_eye_gaze = None
            # if eye_gaze is not None:
            #     seg_eye_gaze = eye_gaze[start_frame:end_frame+1]

            # Decode and resize images to 224x224
            decoded_images = []
            for i in range(num_frames):
                if is_encoded:
                    img = decode_jpeg(seg_images[i])
                else:
                    img = seg_images[i]

                # Resize to 224x224 using PIL
                img_pil = Image.fromarray(img)
                img_resized = img_pil.resize((224, 224), Image.BILINEAR)
                decoded_images.append(np.array(img_resized, dtype=np.uint8))

            # Build episode steps
            episode = []
            for i in range(num_frames):
                obs_dict = {
                    "image": decoded_images[i],
                    "left_ee_pose": seg_left_ee[i].astype(np.float32),
                    "right_ee_pose": seg_right_ee[i].astype(np.float32),
                    "left_keypoints": seg_left_kp[i].astype(np.float32),
                    "right_keypoints": seg_right_kp[i].astype(np.float32),
                    "left_wrist_pose": seg_left_wrist[i].astype(np.float32),
                    "right_wrist_pose": seg_right_wrist[i].astype(np.float32),
                    "head_pose": seg_head[i].astype(np.float32),
                }

                # if seg_eye_gaze is not None:
                #     obs_dict["eye_gaze"] = seg_eye_gaze[i].astype(np.float32)

                episode.append(
                    {
                        "observation": obs_dict,
                        "action": np.zeros(14, dtype=np.float32),  # Placeholder for bimanual actions
                        "discount": 1.0,
                        "reward": float(i == (num_frames - 1)),
                        "is_first": i == 0,
                        "is_last": i == (num_frames - 1),
                        "is_terminal": i == (num_frames - 1),
                        "language_instruction": task_name,
                    }
                )

            # Create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(zarr_path),
                    "recording_name": zarr_path.name,
                    "segment_index": seg_idx,
                    "num_segments": len(task_segments),
                },
            }

            # Return with a unique key
            yield f"{zarr_path.name}_seg{seg_idx}", sample

    # Parse examples from valid paths
    for zarr_path in valid_paths:
        yield from _parse_example(zarr_path)


class AriaDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for Aria bimanual manipulation dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers (reduced to avoid zarr v3 multiprocessing issues)
    MAX_PATHS_IN_MEMORY = 100  # number of paths converted & stored in memory before writing to disk
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="First-person RGB camera observation.",
                                    ),
                                    "left_ee_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Left hand end-effector pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    "right_ee_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Right hand end-effector pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    "left_keypoints": tfds.features.Tensor(
                                        shape=(63,),
                                        dtype=np.float32,
                                        doc="Left hand keypoints (21 points × 3 coordinates). 3D positions in meters.",
                                    ),
                                    "right_keypoints": tfds.features.Tensor(
                                        shape=(63,),
                                        dtype=np.float32,
                                        doc="Right hand keypoints (21 points × 3 coordinates). 3D positions in meters.",
                                    ),
                                    "left_wrist_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Left wrist pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    "right_wrist_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Right wrist pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    "head_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Head/camera pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    # "eye_gaze": tfds.features.Tensor(
                                    #     shape=(3,),
                                    #     dtype=np.float32,
                                    #     doc="Eye gaze direction vector [x, y, z]. 3D normalized direction vector.",
                                    # ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Placeholder for bimanual actions (2 arms × 7D). Currently zeros as this is teleoperation data.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original zarr directory."),
                            "recording_name": tfds.features.Text(doc="Name of the recording (timestamp)."),
                            "segment_index": tfds.features.Scalar(
                                dtype=np.int32, doc="Segment index within recording."
                            ),
                            "num_segments": tfds.features.Scalar(
                                dtype=np.int32, doc="Total number of segments in recording."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Path to aria zarr data
        base_dir = Path("/n/fs/robot-data/EgoVerse/data/aria_fold_clothes")

        if not base_dir.exists():
            print(f"Warning: Dataset path does not exist: {base_dir}")
            return {"train": []}

        # Find all zarr episode directories (timestamped directories)
        # Filter for directories that:
        # 1. Are directories
        # 2. Don't start with '.' (hidden files)
        # 3. Have the timestamp format (YYYY-MM-DD-HH-MM-SS-XXXXXX)
        # 4. Contain a zarr.json file (valid zarr directory)
        zarr_dirs = []
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("."):
                continue
            # Check if it has zarr.json (valid zarr directory)
            if not (d / "zarr.json").exists():
                continue
            zarr_dirs.append(str(d))

        print(f"Found {len(zarr_dirs)} valid zarr recordings in {base_dir}")
        print("Each recording may contain multiple task instances (segmented by timestamps)")

        if len(zarr_dirs) == 0:
            print("Warning: No valid zarr directories found!")

        return {
            "train": zarr_dirs,
        }
