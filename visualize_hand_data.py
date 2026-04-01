#!/usr/bin/env python3
"""
Visualize left.obs_ee_pose, left.obs_keypoints, and left.obs_wrist_pose on video.

Usage:
    python visualize_hand_data.py --zarr-path <path_to_zarr> --output <output.mp4>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import zarr

try:
    import simplejpeg

    HAS_SIMPLEJPEG = True
except ImportError:
    HAS_SIMPLEJPEG = False


def decode_jpeg(data):
    """Decode JPEG data using simplejpeg or cv2 as fallback."""
    # Unwrap nested numpy arrays
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    if HAS_SIMPLEJPEG:
        if isinstance(data, bytes):
            return simplejpeg.decode_jpeg(data)
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            return simplejpeg.decode_jpeg(data.tobytes())

    # Use OpenCV as fallback
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
        arr = data
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    # Convert BGR to RGB since simplejpeg returns RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Hand skeleton edges (from human.py)
FINGER_EDGES = [
    (5, 6),
    (6, 7),
    (7, 0),  # thumb
    (5, 8),
    (8, 9),
    (9, 10),
    (9, 1),  # index
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),  # middle
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),  # ring
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),  # pinky
]

FINGER_COLORS = {
    "thumb": (255, 100, 100),  # red
    "index": (100, 255, 100),  # green
    "middle": (100, 100, 255),  # blue
    "ring": (255, 255, 100),  # yellow
    "pinky": (255, 100, 255),  # magenta
}

FINGER_EDGE_RANGES = [
    ("thumb", 0, 3),
    ("index", 3, 7),
    ("middle", 7, 11),
    ("ring", 11, 15),
    ("pinky", 15, 19),
]


def load_camera_intrinsics(zarr_path):
    """Load camera intrinsics from zarr metadata."""
    # Use zarr.open_group with explicit string path for better zarr v3 compatibility
    store = zarr.open_group(str(zarr_path), mode="r")
    metadata = store.attrs.asdict()

    intrinsics = metadata.get("camera_intrinsics", {})
    if not intrinsics:
        # Default intrinsics for Scale data (640x480)
        intrinsics = {
            "fx": 320.0,
            "fy": 320.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
        }

    return intrinsics


def project_3d_to_2d(points_3d, head_pose, intrinsics):
    """
    Project 3D world points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of xyz coordinates in world frame
        head_pose: (7,) array [x, y, z, qw, qx, qy, qz] - camera pose in world frame
        intrinsics: dict with fx, fy, cx, cy

    Returns:
        (N, 2) array of pixel coordinates, valid_mask (all False if invalid pose)
    """
    # Extract camera pose
    cam_pos = head_pose[:3]
    cam_quat = head_pose[3:]  # [w, x, y, z]

    # Check for valid quaternion (non-zero norm)
    quat_norm = np.linalg.norm(cam_quat)
    if quat_norm < 1e-6:
        # Invalid quaternion - return all invalid
        invalid_pts = np.full((len(points_3d), 2), -1, dtype=np.float32)
        invalid_mask = np.zeros(len(points_3d), dtype=bool)
        return invalid_pts, invalid_mask

    # Convert world points to camera frame
    # T_world_to_cam = inverse of T_cam_to_world
    try:
        R_world_to_cam = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]).as_matrix().T
    except (ValueError, RuntimeError):
        # Invalid rotation - return all invalid
        invalid_pts = np.full((len(points_3d), 2), -1, dtype=np.float32)
        invalid_mask = np.zeros(len(points_3d), dtype=bool)
        return invalid_pts, invalid_mask

    t_world_to_cam = -R_world_to_cam @ cam_pos

    # Transform points to camera frame
    points_cam = (R_world_to_cam @ points_3d.T).T + t_world_to_cam

    # Project to image plane
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    # Avoid division by zero
    z = points_cam[:, 2]
    valid_mask = z > 0.01

    u = np.full(len(points_3d), -1, dtype=np.float32)
    v = np.full(len(points_3d), -1, dtype=np.float32)

    u[valid_mask] = fx * points_cam[valid_mask, 0] / z[valid_mask] + cx
    v[valid_mask] = fy * points_cam[valid_mask, 1] / z[valid_mask] + cy

    return np.stack([u, v], axis=1), valid_mask


def draw_coordinate_frame(image, origin_2d, rotation, intrinsics, scale=0.05, valid=True):
    """Draw a 3D coordinate frame at the given origin."""
    if not valid or origin_2d[0] < 0 or origin_2d[1] < 0:
        return

    # Create axis vectors in local frame
    axes_local = np.eye(3) * scale  # x, y, z axes

    # Rotate to world frame
    axes_world = (rotation @ axes_local.T).T

    # Project to 2D (assuming origin is already in correct frame)
    # For simplicity, just draw from origin_2d
    # This is approximate visualization
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x=red, y=green, z=blue

    origin = tuple(origin_2d.astype(int))

    # Draw axes (approximate 2D projection)
    for i, color in enumerate(colors):
        # Simple 2D approximation - not geometrically correct but shows orientation
        direction = axes_world[i, :2] * 50  # Scale for visibility
        end_point = (int(origin[0] + direction[0]), int(origin[1] + direction[1]))
        cv2.arrowedLine(image, origin, end_point, color, 2, tipLength=0.3)


def visualize_frame(image, left_ee_pose, left_keypoints, left_wrist_pose, head_pose, intrinsics, show_labels=True):
    """
    Visualize hand data on a single frame.

    Args:
        image: (H, W, 3) RGB image
        left_ee_pose: (7,) [x, y, z, qw, qx, qy, qz] - palm pose
        left_keypoints: (63,) or (21, 3) - hand keypoints xyz
        left_wrist_pose: (7,) [x, y, z, qw, qx, qy, qz] - wrist pose
        head_pose: (7,) camera pose
        intrinsics: camera intrinsics dict
        show_labels: whether to show text labels
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Reshape keypoints if needed
    if left_keypoints.shape == (63,):
        keypoints_3d = left_keypoints.reshape(21, 3)
    else:
        keypoints_3d = left_keypoints

    # Project keypoints to 2D
    kp_2d, kp_valid = project_3d_to_2d(keypoints_3d, head_pose, intrinsics)

    # Draw hand skeleton
    for finger_name, start_idx, end_idx in FINGER_EDGE_RANGES:
        color = FINGER_COLORS[finger_name]
        for i in range(start_idx, end_idx):
            edge = FINGER_EDGES[i]
            if kp_valid[edge[0]] and kp_valid[edge[1]]:
                pt1 = tuple(kp_2d[edge[0]].astype(int))
                pt2 = tuple(kp_2d[edge[1]].astype(int))
                # Check if points are within image bounds
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(vis, pt1, pt2, color, 2)

    # Draw keypoints
    for i, (kp, valid) in enumerate(zip(kp_2d, kp_valid)):
        if valid:
            pt = tuple(kp.astype(int))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(vis, pt, 3, (255, 165, 0), -1)

    # Project and draw wrist pose
    wrist_pos = left_wrist_pose[:3].reshape(1, 3)
    wrist_2d, wrist_valid = project_3d_to_2d(wrist_pos, head_pose, intrinsics)

    if wrist_valid[0]:
        wrist_pt = tuple(wrist_2d[0].astype(int))
        if 0 <= wrist_pt[0] < w and 0 <= wrist_pt[1] < h:
            cv2.circle(vis, wrist_pt, 6, (0, 255, 255), 2)  # Cyan circle
            if show_labels:
                cv2.putText(
                    vis, "Wrist", (wrist_pt[0] + 10, wrist_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )

    # Project and draw EE (palm) pose
    ee_pos = left_ee_pose[:3].reshape(1, 3)
    ee_2d, ee_valid = project_3d_to_2d(ee_pos, head_pose, intrinsics)

    if ee_valid[0]:
        ee_pt = tuple(ee_2d[0].astype(int))
        if 0 <= ee_pt[0] < w and 0 <= ee_pt[1] < h:
            cv2.circle(vis, ee_pt, 6, (255, 0, 255), 2)  # Magenta circle
            if show_labels:
                cv2.putText(
                    vis, "Palm (EE)", (ee_pt[0] + 10, ee_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                )

    # Add legend
    if show_labels:
        legend_x, legend_y = 10, 30
        cv2.putText(
            vis, "Left Hand Visualization:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            vis,
            "- Keypoints: Orange dots + colored skeleton",
            (legend_x, legend_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis, "- Wrist: Cyan circle", (legend_x, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
        cv2.putText(
            vis,
            "- Palm (EE): Magenta circle",
            (legend_x, legend_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
        )

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize left hand data (ee_pose, keypoints, wrist_pose) on video")
    parser.add_argument("--zarr-path", required=True, help="Path to zarr episode")
    parser.add_argument("--output", default="hand_visualization.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--no-labels", action="store_true", help="Hide text labels")
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise ValueError(f"Zarr path does not exist: {zarr_path}")

    print(f"Loading data from {zarr_path}...")
    # Use zarr.open_group with explicit string path for better zarr v3 compatibility
    store = zarr.open_group(str(zarr_path), mode="r")

    # Load data
    images_encoded = store["images.front_1"][:]
    left_ee_pose = store["left.obs_ee_pose"][:]
    left_keypoints = store["left.obs_keypoints"][:]
    left_wrist_pose = store["left.obs_wrist_pose"][:]
    head_pose = store["obs_head_pose"][:]

    intrinsics = load_camera_intrinsics(zarr_path)

    T = len(images_encoded)
    print(f"Processing {T} frames...")

    # Decode first image to get shape
    if isinstance(images_encoded[0], bytes):
        first_img = decode_jpeg(images_encoded[0])
    else:
        first_img = images_encoded[0]

    print(f"Image shape: {first_img.shape}")
    print(
        f"Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, "
        f"cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}"
    )

    # Setup video writer
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    # Check if images are encoded
    is_encoded = isinstance(images_encoded[0], bytes)

    invalid_frames = 0
    for t in range(T):
        if t % 30 == 0:
            print(f"Processing frame {t}/{T}...")

        # Decode image if needed
        if is_encoded:
            img = decode_jpeg(images_encoded[t])
        else:
            img = images_encoded[t]

        # Check for invalid head pose
        head_quat_norm = np.linalg.norm(head_pose[t][3:])
        if head_quat_norm < 1e-6:
            if invalid_frames == 0:
                print(f"  Warning: Invalid head pose at frame {t} (and possibly more)")
            invalid_frames += 1
            # Just write the original image without visualization
            vis_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(vis_bgr)
            continue

        vis = visualize_frame(
            img,
            left_ee_pose[t],
            left_keypoints[t],
            left_wrist_pose[t],
            head_pose[t],
            intrinsics,
            show_labels=not args.no_labels,
        )

        # Convert RGB to BGR for OpenCV
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        out.write(vis_bgr)

    out.release()
    print(f"\nVisualization saved to: {args.output}")
    if invalid_frames > 0:
        print(f"Note: {invalid_frames}/{T} frames had invalid head pose (shown without overlay)")


if __name__ == "__main__":
    main()
