"""Aria bimanual human hand dataset implementation.

Aria data is egocentric video of bimanual hand manipulation, captured with
an Aria head-mounted device. Each timestep has:
- A first-person (egocentric) RGB image
- Left and right hand palm poses in the WORLD frame [x, y, z, qw, qx, qy, qz]
- Head (camera) pose in world frame [x, y, z, qw, qx, qy, qz]
- Language instruction

The world frame is fixed at the start of each recording session.

Action representation:
  All future EE poses in a chunk [t+1 .. t+H] are expressed in the semantic
  camera frame at the current timestep t:
    pos_cam_t = R_t^T @ (pos_world - cam_pos_t)
  then a semantic alignment P is applied so that dim0=forward (optical axis),
  dim1=left, dim2=up.  Action deltas are then computed from the current-step
  pose in this semantic frame.  Gripper state is the raw future value (no delta).

  Semantic camera frame (Aria RGB camera: X=right, Y=down, Z=forward):
    P = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

State representation:
  Current left and right hand palm poses expressed in the semantic camera frame
  at each timestep t, as xyz + euler + gripper (14D):
  [left_xyz(3), left_euler(3), left_gripper(1), right_xyz(3), right_euler(3), right_gripper(1)]
  Converted to xyz + R6 + gripper (20D) for training.
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
    - Bimanual: left + right hand, each 7D (xyz + euler + gripper)
    - Egocentric only: primary image is the head camera, no wrist camera
    - Camera-frame relative actions: future EE poses are expressed relative to
      the camera frame at the current timestep before computing deltas;
      gripper is carried as raw future state (no delta)
    - State: 14D (left_xyz + left_euler + left_gripper + right_xyz + right_euler + right_gripper),
             converted to 20D (left_xyz + left_R6 + left_gripper + right_xyz + right_R6 + right_gripper)
    - Gripper: binary (1=open, 0=closed) from left/right_gripper_binary_hybrid
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

    def chunk_actions(
        self,
        traj,
        action_horizon: int,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """Chunk actions with world-to-semantic-camera-frame transformation.

        EE poses (action and state) are in the world frame.  For each timestep t:
          - State: expressed in the semantic camera frame at t.
          - Action chunk [t+1 .. t+H]: all expressed in the semantic camera frame
            at t, then deltas from the current pose are computed.

        Semantic camera frame (Aria RGB: X=right, Y=down, Z=forward):
          P = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
          dim0 = camera +Z  (forward / optical axis)
          dim1 = camera −X  (left)
          dim2 = camera −Y  (up)

        Gripper is carried as the raw future state (no delta), same as robot data.
        """
        traj_len = tf.shape(traj[action_key])[0]

        # Semantic alignment: camera (X=right, Y=down, Z=forward) → (fwd=0, left=1, up=2)
        P  = tf.constant([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], dtype=tf.float32)
        PT = tf.transpose(P)  # P^T for similarity transforms on rotation matrices

        # ── Per-timestep camera geometry ──────────────────────────────────────
        # head_pose: [T, 7] = [x, y, z, qw, qx, qy, qz]  (camera in world frame)
        head_pos_all   = traj["head_pose"][:, :3]   # [T, 3]
        hq_wxyz        = traj["head_pose"][:, 3:7]  # [T, 4]
        hq_xyzw        = tf.concat([hq_wxyz[:, 1:4], hq_wxyz[:, 0:1]], axis=-1)
        R_c2w          = quaternion_to_rotation_matrix(hq_xyzw)  # [T, 3, 3]  cam→world

        # For windowed ops, broadcast the reference-frame rotation/translation
        R_t = R_c2w[:, None, :, :]       # [T, 1, 3, 3]  broadcast over window dim
        t_t = head_pos_all[:, None, :]   # [T, 1, 3]

        # ── Transform state: world → semantic cam frame at each t ─────────────
        def _to_semantic_per_t(pos_w, euler_w):
            """[T,3], [T,3] → pos_s [T,3], euler_s [T,3] in semantic cam frame."""
            # pos_cam = R_c2w^T @ (pos_world − cam_pos)
            pos_cam  = tf.einsum("tji,tj->ti", R_c2w, pos_w - head_pos_all)
            pos_s    = tf.einsum("ij,tj->ti",  P,     pos_cam)

            R_ee_w   = euler_to_rotation_matrix(euler_w)              # [T,3,3]
            R_ee_cam = tf.einsum("tji,tjk->tik", R_c2w, R_ee_w)      # R^T @ R_ee_w
            R_ee_s   = tf.einsum("ij,tjk,kl->til", P, R_ee_cam, PT)  # P @ R_ee_cam @ P^T
            return pos_s, rotation_matrix_to_euler(R_ee_s)

        def _transform_state(s14):
            lp, le = _to_semantic_per_t(s14[:, :3],   s14[:, 3:6])
            rp, re = _to_semantic_per_t(s14[:, 7:10], s14[:, 10:13])
            return tf.concat([lp, le, s14[:, 6:7], rp, re, s14[:, 13:14]], axis=-1)

        traj["observation"][state_key] = _transform_state(traj["observation"][state_key])
        traj["raw_state"]              = _transform_state(traj["raw_state"])

        # ── Gather sliding windows of world-frame EE poses ────────────────────
        windowed_actions = gather_with_last_value_padding(
            data=traj[action_key],
            sequence_length=traj_len,
            window_size=action_horizon + 1,
        )  # [T, W, 14]

        # ── Transform windowed poses: world → semantic cam frame at t ─────────
        def _to_semantic_windowed(ee_xyz_euler):
            """[T,W,6] world frame → [T,W,6] semantic cam frame at t."""
            pos_w   = ee_xyz_euler[..., :3]   # [T, W, 3]
            euler_w = ee_xyz_euler[..., 3:6]  # [T, W, 3]

            # pos_cam_t = R_t^T @ (pos_world − t_t);  R_t broadcasts over W
            pos_cam_t = tf.einsum("...ji,...j->...i", R_t, pos_w - t_t)
            pos_s     = tf.einsum("ij,...j->...i",    P,   pos_cam_t)

            R_ee_w   = euler_to_rotation_matrix(euler_w)               # [T,W,3,3]
            R_ee_cam = tf.einsum("...ji,...jk->...ik", R_t,  R_ee_w)  # R_t^T @ R_ee_w
            R_ee_s   = tf.einsum("ij,...jk,kl->...il", P, R_ee_cam, PT)
            return tf.concat([pos_s, rotation_matrix_to_euler(R_ee_s)], axis=-1)

        # action layout: [l_xyz(3), l_euler(3), l_grip(1), r_xyz(3), r_euler(3), r_grip(1)]
        left_ee_w  = windowed_actions[..., :6]     # [T, W, 6]
        left_grip  = windowed_actions[..., 6:7]    # [T, W, 1]
        right_ee_w = windowed_actions[..., 7:13]   # [T, W, 6]
        right_grip = windowed_actions[..., 13:14]  # [T, W, 1]

        left_s  = _to_semantic_windowed(left_ee_w)   # [T, W, 6]
        right_s = _to_semantic_windowed(right_ee_w)  # [T, W, 6]

        # Deltas: future steps (1..H) minus current step (0) in semantic cam frame
        left_delta_pos   = left_s[:, 1:, :3] - left_s[:, 0:1, :3]              # [T,H,3]
        left_delta_euler = euler_diff(left_s[:, 1:, 3:6], left_s[:, 0:1, 3:6]) # [T,H,3]
        right_delta_pos   = right_s[:, 1:, :3] - right_s[:, 0:1, :3]           # [T,H,3]
        right_delta_euler = euler_diff(right_s[:, 1:, 3:6], right_s[:, 0:1, 3:6])

        left_grip_chunk  = left_grip[:, :-1, :]   # [T, H, 1]
        right_grip_chunk = right_grip[:, :-1, :]  # [T, H, 1]

        # Store per-timestep "world → semantic cam" rotation for group_language_actions.
        # lang_R[t] = P @ R_c2w[t]^T  maps a world-frame vector to semantic cam frame at t.
        R_world_to_sem = tf.einsum("ij,tkj->tik", P, R_c2w)  # [T, 3, 3]
        traj["lang_R"] = tf.reshape(R_world_to_sem, [-1, 9])  # [T, 9]

        # Store world-frame EE orientations for the rotation-frame transform.
        # Use windowed_actions[:, 0, :] — window index 0 is the current timestep t
        # in world-frame coordinates, identical to traj[action_key][t] before it is
        # overwritten below.  Extracting here avoids reading the wrong tensor.
        # action layout: [l_xyz(3), l_euler(3), l_grip(1), r_xyz(3), r_euler(3), r_grip(1)]
        R_ee_left_w  = euler_to_rotation_matrix(windowed_actions[:, 0, 3:6])    # [T, 3, 3]
        R_ee_right_w = euler_to_rotation_matrix(windowed_actions[:, 0, 10:13])  # [T, 3, 3]
        traj["lang_ee_R_left"]  = tf.reshape(R_ee_left_w,  [-1, 9])  # [T, 9]
        traj["lang_ee_R_right"] = tf.reshape(R_ee_right_w, [-1, 9])  # [T, 9]

        # Final action chunks: [T, action_horizon, 14]
        # Format: [l_dxyz(3), l_deuler(3), l_grip(1), r_dxyz(3), r_deuler(3), r_grip(1)]
        traj[action_key] = tf.concat(
            [
                left_delta_pos, left_delta_euler, left_grip_chunk,
                right_delta_pos, right_delta_euler, right_grip_chunk,
            ],
            axis=-1,
        )

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
        For bimanual 14D state [l_xyz(3), l_euler(3), l_gripper(1),
                                  r_xyz(3), r_euler(3), r_gripper(1)],
        we also need to convert [10:13] (right euler) to R6, keeping gripper dims.
        Result: 20D = [l_xyz(3), l_R6(6), l_gripper(1), r_xyz(3), r_R6(6), r_gripper(1)].
        """

        # Step 1: World → semantic camera frame transformation + action chunking.
        # chunk_actions uses head_pose (then removes it) and outputs the state in
        # euler format so that Step 2 can convert to R6.
        self.dataset = self.dataset.traj_map(
            lambda traj: self.chunk_actions(
                traj,
                action_horizon=action_horizon,
                action_key=action_key,
                state_key=state_key,
            ),
            self.num_parallel_calls,
        )

        # Step 2: Convert both hands' euler to R6 (must run after chunk_actions so
        # the state is already in the semantic camera frame, still in euler format).
        def bimanual_state_euler_to_rot6d(traj):
            def convert(s):
                # s: [T, 14] = [l_xyz(3), l_euler(3), l_gripper(1),
                #               r_xyz(3), r_euler(3), r_gripper(1)]
                return tf.concat(
                    [
                        s[:, :3],                    # left xyz
                        euler_to_rot6d(s[:, 3:6]),   # left R6
                        s[:, 6:7],                   # left gripper
                        s[:, 7:10],                  # right xyz
                        euler_to_rot6d(s[:, 10:13]), # right R6
                        s[:, 13:14],                 # right gripper
                    ],
                    axis=-1,
                )  # [T, 20]

            traj["observation"][state_key] = convert(traj["observation"][state_key])
            traj["raw_state"] = convert(traj["raw_state"])
            return traj

        self.dataset = self.dataset.traj_map(
            bimanual_state_euler_to_rot6d, self.num_parallel_calls
        )

        # Step 3: Pad actions, language_action, and state to fixed global dimensions
        def pad_action_state(traj):
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(
                traj[action_key],
                [[0, 0], [0, 0], [0, pad_amount_action]],
            )
            traj[action_key].set_shape([None, action_horizon, self.action_dim])

            # Pad language_action to action_dim so all datasets have the same
            # last dimension when mixed in a batch (single-arm 7D, bimanual 14D).
            lang_action_last_dim = tf.shape(traj["language_action"])[-1]
            pad_amount_lang = tf.maximum(0, self.action_dim - lang_action_last_dim)
            traj["language_action"] = tf.pad(
                traj["language_action"],
                [[0, 0], [0, pad_amount_lang]],
            )
            traj["language_action"].set_shape([None, self.action_dim])

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

        # Shared helper: rotate both translation and rotation dims to semantic cam frame.
        # layout: [l_xyz(3), l_euler(3), l_grip(1), r_xyz(3), r_euler(3), r_grip(1)]
        #
        # Translation (0:3, 7:10):
        #   lang_R[t] @ Σ Δpos_world  →  semantic cam frame at t
        #
        # Rotation (3:6, 10:13):
        #   sum_actions composes euler_diff steps, telescoping to:
        #     R_total_body = R_world_ee[t]^T @ R_world_ee[t+H]  (EE body frame at t)
        #   We want it in semantic cam frame at t:
        #     Q = lang_R @ R_world_ee[t]   (EE body frame → semantic cam frame)
        #     R_total_sem = Q @ R_total_body @ Q^T
        #   This aligns axes: tilt fwd/bwd ≈ rot around X, left/right ≈ Y, CW/CCW ≈ Z.
        def _rotate_lang_trans(actions_14d, rot_3x3, r_ee_left, r_ee_right):
            """Rotate translation and rotation dims from world/body frame to semantic cam.

            Args:
                actions_14d: [T, 14] summed language actions (world-frame translations,
                             body-frame rotations from sum_actions telescoping).
                rot_3x3:    [T, 3, 3]  lang_R = P @ R_c2w^T
                r_ee_left:  [T, 3, 3]  world-frame left-EE orientation at t
                r_ee_right: [T, 3, 3]  world-frame right-EE orientation at t
            Returns:
                [T, 14] with all translation and rotation dims in semantic cam frame.
            """
            # ── Translation ───────────────────────────────────────────────────
            left_xyz  = tf.einsum("tij,tj->ti", rot_3x3, actions_14d[:, :3])    # [T, 3]
            right_xyz = tf.einsum("tij,tj->ti", rot_3x3, actions_14d[:, 7:10])  # [T, 3]

            # ── Rotation ──────────────────────────────────────────────────────
            # Q = lang_R @ R_world_ee[t]  transforms from EE body frame to semantic cam
            q_left  = tf.einsum("tij,tjk->tik", rot_3x3, r_ee_left)   # [T, 3, 3]
            q_right = tf.einsum("tij,tjk->tik", rot_3x3, r_ee_right)  # [T, 3, 3]

            r_body_left  = euler_to_rotation_matrix(actions_14d[:, 3:6])    # [T, 3, 3]
            r_body_right = euler_to_rotation_matrix(actions_14d[:, 10:13])  # [T, 3, 3]

            # R_sem = Q @ R_body @ Q^T  (similarity transform)
            r_sem_left  = tf.einsum("tij,tjk,tlk->til", q_left,  r_body_left,  q_left)
            r_sem_right = tf.einsum("tij,tjk,tlk->til", q_right, r_body_right, q_right)

            euler_sem_left  = rotation_matrix_to_euler(r_sem_left)   # [T, 3]
            euler_sem_right = rotation_matrix_to_euler(r_sem_right)  # [T, 3]

            return tf.concat(
                [
                    left_xyz,                # 0:3   left xyz   (semantic cam)
                    euler_sem_left,          # 3:6   left euler (semantic cam)
                    actions_14d[:, 6:7],     # 6:7   left gripper
                    right_xyz,               # 7:10  right xyz  (semantic cam)
                    euler_sem_right,         # 10:13 right euler (semantic cam)
                    actions_14d[:, 13:],     # 13:14 right gripper
                ],
                axis=-1,
            )

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

            # Sum world-frame per-step deltas over the chosen horizon, then rotate.
            # Σᵢ Δpos_world[t+i] = pos_world[t+H] − pos_world[t]  (world-frame total)
            # lang_R[t] @ total = P @ R_c2w[t]^T @ total  (semantic cam frame at t)
            language_actions = sum_actions(actions_window, valid_lengths)
            lang_r     = tf.reshape(traj["lang_R"],         [-1, 3, 3])
            r_ee_left  = tf.reshape(traj["lang_ee_R_left"], [-1, 3, 3])
            r_ee_right = tf.reshape(traj["lang_ee_R_right"],[-1, 3, 3])
            language_actions = _rotate_lang_trans(
                language_actions, lang_r, r_ee_left, r_ee_right
            )

            traj["language_actions"] = language_actions
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

            pred_lang  = sum_actions(actions_window, deltas)
            lang_r     = tf.reshape(traj["lang_R"],         [-1, 3, 3])
            r_ee_left  = tf.reshape(traj["lang_ee_R_left"], [-1, 3, 3])
            r_ee_right = tf.reshape(traj["lang_ee_R_right"],[-1, 3, 3])
            pred_lang  = _rotate_lang_trans(
                pred_lang, lang_r, r_ee_left, r_ee_right
            )

            pred_lang.set_shape([None, traj["language_action"].shape[-1]])
            traj["prediction_language_actions"] = pred_lang
            traj["prediction_delta"] = deltas

            return traj

        self.dataset = self.dataset.traj_map(
            add_prediction_pairs, self.num_parallel_calls
        )

        # Cleanup: remove pipeline-internal fields consumed above.
        _aux_keys = {"lang_R", "lang_ee_R_left", "lang_ee_R_right"}
        self.dataset = self.dataset.traj_map(
            lambda traj: {k: v for k, v in traj.items() if k not in _aux_keys},
            self.num_parallel_calls,
        )
