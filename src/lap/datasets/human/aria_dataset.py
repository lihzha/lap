"""Aria bimanual human hand dataset implementation.

Aria data is egocentric video of bimanual hand manipulation, captured with
an Aria head-mounted device. Each timestep has:
- A first-person (egocentric) RGB image
- Left and right hand palm poses in the camera frame [x, y, z, qw, qx, qy, qz]
- Head (camera) pose in world frame [x, y, z, qw, qx, qy, qz]
- Language instruction

Action representation:
  For each future timestep t+i, the EE pose is first expressed in the camera
  frame at the current timestep t via:
    a^H_{t+i} = (T_t^device)^{-1} * T_{t+i}^device * p^H_{t+i}
  Then action deltas are computed from this camera-frame reference pose at t.

State representation:
  Current left and right hand palm poses in the camera frame, as xyz + euler (12D).
  Converted to xyz + R6 (18D) for training.
"""

from __future__ import annotations

import tensorflow as tf

from lap.datasets.base_dataset import sum_actions
from lap.datasets.output_schema import TrajectoryOutputBuilder
from lap.datasets.registry import register_dataset
from lap.datasets.robot.oxe_datasets import SingleOXEDataset
from lap.datasets.utils.rotation_utils import euler_diff
from lap.datasets.utils.rotation_utils import euler_to_r6 as euler_to_rot6d
from lap.datasets.utils.rotation_utils import euler_to_rotation_matrix
from lap.datasets.utils.rotation_utils import quaternion_to_rotation_matrix
from lap.datasets.utils.rotation_utils import rotation_matrix_to_euler
from lap.datasets.utils.tfdata_pipeline import gather_with_last_value_padding
from lap.datasets.utils.tfdata_pipeline import gather_with_padding


@register_dataset(matcher=lambda n: "aria" in n, priority=10)
class AriaDataset(SingleOXEDataset):
    """Dataset for Aria bimanual human egocentric manipulation data.

    Differences from single-arm robot datasets:
    - Bimanual: left + right hand, each 6D pose (xyz + euler), no gripper
    - Egocentric only: primary image is the head camera, no wrist camera
    - Camera-frame relative actions: future EE poses are expressed relative to
      the camera frame at the current timestep before computing deltas
    - State: 12D (left_xyz + left_euler + right_xyz + right_euler),
             converted to 18D (left_xyz + left_R6 + right_xyz + right_R6) for training
    """

    FORCE_NO_WRIST_IMAGE: bool = True

    def apply_restructure(self):
        """Restructure trajectory, preserving head_pose for use in chunk_actions."""

        def restructure(traj):
            traj_len = tf.shape(traj["action"])[0]

            # Build observation: maps "image" -> primary_image_key, state from "state" key
            new_obs = self._build_observation(traj, traj_len)

            return TrajectoryOutputBuilder.build_robot_trajectory(
                observation=new_obs,
                actions=traj["action"],
                language_action=traj["language_action"],
                prompt=traj["language_instruction"],
                trajectory_id=traj["trajectory_id"],
                dataset_name=self.dataset_name,
                traj_len=traj_len,
                is_bimanual=True,
                state_type="eef_pose",
                has_wrist_image=False,
                needs_wrist_rotation=False,
                is_navigation=False,
                # head_pose is needed by chunk_actions; carry it through the pipeline
                extra_fields={"head_pose": traj["head_pose"]},
            )

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def chunk_actions(self, traj, action_horizon: int, action_key: str = "actions"):
        """Chunk actions with camera-frame relative transformation.

        For each timestep t and each future step i in [1, action_horizon]:
          1. Express p^H_{t+i} (EE pose in cam frame at t+i) in world frame using head_pose_{t+i}
          2. Re-express in cam frame at t using head_pose_t
          3. Compute delta from current cam-frame pose at t

        This implements:
          a^H_{t:t+k} = [(T_t^device)^{-1} * T_{t+i}^device * p^H_{t+i}]_{i=1}^k
        followed by delta computation (position subtraction + euler_diff).
        """
        traj_len = tf.shape(traj[action_key])[0]

        # Gather sliding windows of absolute EE poses (xyz+euler, 12D)
        # Shape: [T, action_horizon+1, 12]
        windowed_actions = gather_with_last_value_padding(
            data=traj[action_key],
            sequence_length=traj_len,
            window_size=action_horizon + 1,
        )

        # Gather sliding windows of head poses [x,y,z,qw,qx,qy,qz]
        # Shape: [T, action_horizon+1, 7]
        windowed_head_poses = gather_with_last_value_padding(
            data=traj["head_pose"],
            sequence_length=traj_len,
            window_size=action_horizon + 1,
        )

        # Build rotation matrices R_cam_to_world for each (timestep, window) pair
        # head_pose format: [x, y, z, qw, qx, qy, qz]
        head_pos = windowed_head_poses[..., :3]         # [T, W, 3]
        head_quat_wxyz = windowed_head_poses[..., 3:7]  # [T, W, 4]
        # Convert wxyz -> xyzw for quaternion_to_rotation_matrix
        head_quat_xyzw = tf.concat(
            [head_quat_wxyz[..., 1:4], head_quat_wxyz[..., 0:1]], axis=-1
        )
        R_cam_to_world = quaternion_to_rotation_matrix(head_quat_xyzw)  # [T, W, 3, 3]

        # Reference frame: camera at timestep t (window index 0)
        R_t = R_cam_to_world[:, 0:1, :, :]  # [T, 1, 3, 3]  (R_cam_t -> world)
        t_t = head_pos[:, 0:1, :]            # [T, 1, 3]     (translation of cam_t in world)

        def transform_to_cam_t(ee_xyz_euler):
            """Re-express EE poses from their per-step camera frames into cam frame at t.

            Args:
                ee_xyz_euler: [T, W, 6] EE poses in camera frame at each t+i (xyz + euler)

            Returns:
                [T, W, 6] EE poses expressed in camera frame at time t
            """
            pos_cam = ee_xyz_euler[..., :3]    # [T, W, 3]
            euler_cam = ee_xyz_euler[..., 3:6]  # [T, W, 3]

            # EE rotation matrix in cam frame at t+i
            R_ee_cam = euler_to_rotation_matrix(euler_cam)  # [T, W, 3, 3]

            # Transform EE position to world frame:
            #   pos_world = R_cam_to_world @ pos_cam + t_cam
            pos_world = (
                tf.einsum("...ij,...j->...i", R_cam_to_world, pos_cam) + head_pos
            )  # [T, W, 3]

            # Transform EE rotation to world frame:
            #   R_ee_world = R_cam_to_world @ R_ee_cam
            R_ee_world = tf.einsum(
                "...ij,...jk->...ik", R_cam_to_world, R_ee_cam
            )  # [T, W, 3, 3]

            # Transform EE position to cam_t frame:
            #   pos_cam_t = R_t^T @ (pos_world - t_t)
            # Using einsum with swapped indices for transpose: 'ji' instead of 'ij'
            pos_cam_t = tf.einsum(
                "...ji,...j->...i", R_t, pos_world - t_t
            )  # [T, W, 3]

            # Transform EE rotation to cam_t frame:
            #   R_ee_cam_t = R_t^T @ R_ee_world
            R_ee_cam_t = tf.einsum(
                "...ji,...jk->...ik", R_t, R_ee_world
            )  # [T, W, 3, 3]

            euler_cam_t = rotation_matrix_to_euler(R_ee_cam_t)  # [T, W, 3]

            return tf.concat([pos_cam_t, euler_cam_t], axis=-1)  # [T, W, 6]

        # Split and transform each hand
        left_cam_t = transform_to_cam_t(windowed_actions[..., :6])   # [T, W, 6]
        right_cam_t = transform_to_cam_t(windowed_actions[..., 6:12])  # [T, W, 6]

        # Compute deltas: pose at t+i (indices 1..H) minus pose at t (index 0)
        left_delta_pos = left_cam_t[:, 1:, :3] - left_cam_t[:, 0:1, :3]   # [T, H, 3]
        left_delta_euler = euler_diff(
            left_cam_t[:, 1:, 3:6], left_cam_t[:, 0:1, 3:6]
        )  # [T, H, 3]

        right_delta_pos = right_cam_t[:, 1:, :3] - right_cam_t[:, 0:1, :3]  # [T, H, 3]
        right_delta_euler = euler_diff(
            right_cam_t[:, 1:, 3:6], right_cam_t[:, 0:1, 3:6]
        )  # [T, H, 3]

        # Final action chunks: [T, action_horizon, 12]
        # Format: [left_dxyz, left_deuler, right_dxyz, right_deuler]
        traj[action_key] = tf.concat(
            [left_delta_pos, left_delta_euler, right_delta_pos, right_delta_euler],
            axis=-1,
        )

        # head_pose is no longer needed after action chunking
        traj.pop("head_pose", None)

        return traj

    def apply_traj_transforms(
        self,
        action_horizon,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """Override to handle bimanual state correctly.

        The standard state_euler_to_rot6d only converts indices [3:6] (left euler).
        For bimanual 12D state [l_xyz(3), l_euler(3), r_xyz(3), r_euler(3)],
        we also need to convert [9:12] (right euler) to R6.
        Result: 18D = [l_xyz(3), l_R6(6), r_xyz(3), r_R6(6)].
        """

        # Step 1: Convert both hands' euler to R6 for continuous rotation representation
        def bimanual_state_euler_to_rot6d(traj):
            def convert(s):
                # s: [T, 12] = [l_xyz(3), l_euler(3), r_xyz(3), r_euler(3)]
                return tf.concat(
                    [
                        s[:, :3],                   # left xyz
                        euler_to_rot6d(s[:, 3:6]),  # left R6
                        s[:, 6:9],                  # right xyz
                        euler_to_rot6d(s[:, 9:12]), # right R6
                    ],
                    axis=-1,
                )  # [T, 18]

            traj["observation"][state_key] = convert(traj["observation"][state_key])
            traj["raw_state"] = convert(traj["raw_state"])
            return traj

        self.dataset = self.dataset.traj_map(
            bimanual_state_euler_to_rot6d, self.num_parallel_calls
        )

        # Step 2: Camera-frame relative action chunking (uses and removes head_pose)
        self.dataset = self.dataset.traj_map(
            lambda traj: self.chunk_actions(
                traj, action_horizon=action_horizon, action_key=action_key
            ),
            self.num_parallel_calls,
        )

        # Step 3: Pad actions and state to fixed global dimensions
        def pad_action_state(traj):
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(
                traj[action_key],
                [[0, 0], [0, 0], [0, pad_amount_action]],
            )
            traj[action_key].set_shape([None, action_horizon, self.action_dim])

            state_last_dim = tf.shape(traj["observation"][state_key])[-1]
            pad_amount_state = tf.maximum(0, self.state_dim - state_last_dim)
            traj["observation"][state_key] = tf.pad(
                traj["observation"][state_key],
                [[0, 0], [0, pad_amount_state]],
            )
            traj["observation"][state_key].set_shape([None, self.state_dim])

            raw_state_last_dim = tf.shape(traj["raw_state"])[-1]
            pad_amount_raw_state = tf.maximum(0, self.state_dim - raw_state_last_dim)
            traj["raw_state"] = tf.pad(
                traj["raw_state"],
                [[0, 0], [0, pad_amount_raw_state]],
            )
            traj["raw_state"].set_shape([None, self.state_dim])

            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        # Step 4: Group language actions over variable horizon windows
        def group_language_actions(traj):
            traj_len = tf.shape(traj[action_key])[0]
            timestep_ids = tf.range(traj_len, dtype=tf.int32)
            remaining = tf.maximum(traj_len - timestep_ids, 1)

            horizon_seconds = tf.constant(self.horizon_seconds, dtype=tf.float32)
            control_freq = tf.cast(self.control_frequency, tf.float32)
            horizon_steps = tf.cast(tf.round(horizon_seconds * control_freq), tf.int32)
            horizon_steps = tf.maximum(horizon_steps, 1)

            traj_id_hash = tf.strings.to_hash_bucket_fast(
                traj["trajectory_id"][0], 2147483647
            )
            seed_pair = [self.seed, traj_id_hash]
            horizon_idx = tf.random.stateless_uniform(
                [traj_len],
                seed=seed_pair,
                minval=0,
                maxval=tf.shape(horizon_steps)[0],
                dtype=tf.int32,
            )
            chosen_steps = tf.gather(horizon_steps, horizon_idx)
            valid_lengths = tf.minimum(chosen_steps, remaining)

            max_window = tf.reduce_max(horizon_steps)
            actions_window = gather_with_padding(
                data=traj["language_action"],
                sequence_length=traj_len,
                window_size=max_window,
                per_timestep_windows=chosen_steps,
            )

            traj["language_actions"] = sum_actions(actions_window, valid_lengths)
            traj["language_actions"].set_shape(
                [None, traj["language_action"].shape[-1]]
            )

            actual_horizon_seconds = tf.cast(valid_lengths, tf.float32) / control_freq
            traj["time_horizon_seconds"] = actual_horizon_seconds
            traj["time_horizon_seconds"].set_shape([None])

            return traj

        self.dataset = self.dataset.traj_map(
            group_language_actions, self.num_parallel_calls
        )

        # Step 5: Add prediction pairs for prediction training (if enabled)
        def add_prediction_pairs(traj):
            if not self.enable_prediction_training:
                return traj

            traj_len = tf.shape(traj[action_key])[0]
            max_horizon = int(2.5 * self.control_frequency)
            max_horizon_clamped = tf.minimum(max_horizon, traj_len - 1)
            max_horizon_clamped = tf.maximum(max_horizon_clamped, 1)

            traj_id_hash = tf.strings.to_hash_bucket_fast(
                traj["trajectory_id"][0], 2147483647
            )
            seed_pair = [self.seed, traj_id_hash]

            deltas = tf.random.stateless_uniform(
                [traj_len],
                seed=seed_pair,
                minval=max_horizon_clamped,
                maxval=max_horizon_clamped + 1,
                dtype=tf.int32,
            )
            future_indices = tf.minimum(
                tf.range(traj_len, dtype=tf.int32) + deltas, traj_len - 1
            )

            current_imgs = traj["observation"][self.spec.primary_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.primary_image_key] = tf.stack(
                [current_imgs, future_imgs], axis=1
            )

            current_imgs = traj["observation"][self.spec.wrist_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.wrist_image_key] = tf.stack(
                [current_imgs, future_imgs], axis=1
            )

            actions_window = gather_with_padding(
                data=traj["language_action"],
                sequence_length=traj_len,
                window_size=max_horizon,
                per_timestep_windows=deltas,
            )

            prediction_lang_actions = sum_actions(actions_window, deltas)
            prediction_lang_actions.set_shape(
                [None, traj["language_action"].shape[-1]]
            )

            traj["prediction_language_actions"] = prediction_lang_actions
            traj["prediction_delta"] = deltas

            return traj

        self.dataset = self.dataset.traj_map(
            add_prediction_pairs, self.num_parallel_calls
        )
