import dataclasses
import datetime
import faulthandler
import os
import sys
import time

import cv2
import h5py
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

sys.path.append(".")
from helpers import euler_to_rot6d
from helpers import interpolate_rpy
from helpers import prevent_keyboard_interrupt
from helpers import ActionChunkPostProcessor

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "31177322"  # e.g., "24259877"
    right_camera_id: str = "38872458"  # e.g., "24514023"
    wrist_camera_id: str = "10501775"  # e.g., "13062452"
    # Policy parameters
    external_camera: str = "right"  # which external camera should be fed to the policy, choose from ["left", "right"]
    # Rollout parameters
    max_timesteps: int = 1200
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8
    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )
    use_wrist_camera: bool = True  # whether to use the wrist camera image as input to the policy
    run_upstream: bool = False  # whether to run the upstream policy from OpenPI
    # Rollout saving parameters
    save_rollout: bool = False  # whether to save rollout data to h5 file
    rollout_save_dir: str = "./rollouts"  # directory to save rollout h5 files
    rollout_subsample_rate: int = 10  # subsample rate when saving rollout (1 = no subsampling)


class BaseEvalRunner:
    CHUNK_STEPS = 8

    def __init__(self, args):
        self.env = self.init_env()
        self.args: Args = args
        self.side_image_name = None
        self.action_postprocessor = ActionChunkPostProcessor(
            chunk_steps=self.CHUNK_STEPS,
            use_quaternions=self.use_quaternion_actions(),
        )

    def init_env(self):
        from droid.robot_env import RobotEnv

        return RobotEnv(
            action_space="cartesian_position",
            gripper_action_space="position",
        )

    def binarize_gripper(self, action):
        # Binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])
        return action

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["image"]
        robot_state = obs_dict["robot_state"]

        side_key, wrist_key = self.resolve_camera_keys(image_observations)
        side_image = self._process_camera_image(
            image_observations[side_key],
            rotate_180=self.side_camera_rotate_180(),
        )
        wrist_image = self._process_camera_image(
            image_observations[wrist_key],
            rotate_180=self.wrist_camera_rotate_180(),
        )

        if save_to_disk:
            self._save_camera_preview(side_image, wrist_image)

        raw_cartesian = np.asarray(robot_state["cartesian_position"])
        euler = raw_cartesian[3:6].copy()
        cartesian_position = self.process_cartesian_position(raw_cartesian, euler)
        gripper_position = self.process_gripper_observation(np.array([robot_state["gripper_position"]]))

        return {
            "right_image": side_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "joint_position": np.asarray(robot_state["joint_positions"]),
            "gripper_position": np.asarray(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "euler": euler,
        }

    def resolve_camera_keys(self, image_observations):
        raise NotImplementedError()

    def find_camera_key(self, image_observations, camera_id: str, *, require_left_stream: bool = True) -> str:
        for key in image_observations:
            if camera_id in key and (not require_left_stream or "left" in key):
                return key
        stream_hint = " with 'left' stream" if require_left_stream else ""
        raise KeyError(f"Could not resolve camera key for id '{camera_id}'{stream_hint}.")

    def side_camera_rotate_180(self) -> bool:
        return False

    def wrist_camera_rotate_180(self) -> bool:
        return True

    def use_rot6d_state(self) -> bool:
        return True

    def process_cartesian_position(self, cartesian_position: np.ndarray, euler: np.ndarray) -> np.ndarray:
        if self.use_rot6d_state():
            return np.concatenate([cartesian_position[:3], euler_to_rot6d(euler)])
        return cartesian_position

    def process_gripper_observation(self, gripper_position: np.ndarray) -> np.ndarray:
        return gripper_position

    def _process_camera_image(self, image: np.ndarray, *, rotate_180: bool) -> np.ndarray:
        image = np.asarray(image)[..., :3]
        image = image[..., ::-1]
        if rotate_180:
            image = image[::-1, ::-1]
        return image_tools.resize_with_pad(image, 224, 224)

    def _save_camera_preview(self, side_image: np.ndarray, wrist_image: np.ndarray) -> None:
        from PIL import Image

        combined_image = np.concatenate([side_image, wrist_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    def use_quaternion_actions(self) -> bool:
        """Override for robots/environments that execute quaternion actions."""
        return False

    def _record_frames(self, curr_obs, video, wrist_video):
        side_frame = curr_obs.get(self.side_image_name)
        if side_frame is None and self.args.external_camera is not None:
            side_frame = curr_obs.get(f"{self.args.external_camera}_image")
        if side_frame is not None:
            video.append(side_frame[0] if len(side_frame.shape) == 4 else side_frame)

        wrist_frame = curr_obs.get("wrist_image")
        if wrist_frame is not None:
            wrist_video.append(wrist_frame[0].copy() if len(wrist_frame.shape) == 4 else wrist_frame.copy())

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation":{
                "base_0_rgb": curr_obs[self.side_image_name],
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
                "euler": curr_obs["euler"],
            },
            "prompt": instruction,
        }
        if self.args.use_wrist_camera and "wrist_image" in curr_obs:
            request["observation"]["left_wrist_0_rgb"] = curr_obs["wrist_image"]
        return request

    def process_gripper_action(self, action, curr_obs):
        return 1 - curr_obs["gripper_position"] if len(action) == 6 else 1 - action[..., -1]

    def get_action_from_response(self, response, curr_obs):
        curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
        curr_rpy = np.asarray(curr_obs["euler"], dtype=float)
        return self.action_postprocessor.process_response(
            response,
            curr_pos=curr_pos,
            curr_rpy=curr_rpy,
            gripper_postprocess_fn=lambda action: self.process_gripper_action(action, curr_obs),
            orientation_interpolator_fn=interpolate_rpy,
        )

    @staticmethod
    def _slugify(text: str) -> str:
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
        return sanitized or "unnamed"

    def _concat_and_save_video(self, instruction, video, wrist_video, policy_name=None, success_label=None):
        assert video is not None, "Side video stream should always be recorded."
        if not video:
            return
        video = np.stack(video)
        combined_video = video
        if wrist_video:
            wrist_video = np.stack(wrist_video)

            # Ensure both videos have the same width by resizing wrist view to match side view width
            _, side_width = video.shape[1:3]
            wrist_height, wrist_width = wrist_video.shape[1:3]

            if wrist_width != side_width:
                resized_wrist = []
                aspect_ratio = wrist_width / wrist_height
                new_height = int(side_width / aspect_ratio)
                for frame in wrist_video:
                    resized_frame = cv2.resize(frame, (side_width, new_height))
                    resized_wrist.append(resized_frame)
                wrist_video = np.stack(resized_wrist)

            # Concatenate side view and wrist view vertically (wrist below side view)
            combined_video = np.concatenate([video, wrist_video], axis=1)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        filename_parts = [
            "video",
            self._slugify(policy_name) if policy_name else None,
            self._slugify(instruction),
            self._slugify(success_label) if success_label else None,
            timestamp,
        ]
        save_filename = "_".join(part for part in filename_parts if part)
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", fps=10, codec="libx264")

    def _save_rollout_to_h5(self, rollout_data, instruction, policy_name=None, success_label=None):
        """Save rollout data to h5 file.

        Args:
            rollout_data: dict with keys:
                - 'cartesian_positions': list of (6,) arrays
                - 'gripper_positions': list of floats
                - 'actions': list of (7,) arrays (cartesian position + gripper)
                - 'side_images': list of image arrays
                - 'wrist_images': list of image arrays (optional)
            instruction: str
            policy_name: str or None
            success_label: str or None
        """
        if not rollout_data["actions"]:
            print("No actions recorded, skipping h5 save.")
            return

        # Create save directory if it doesn't exist
        os.makedirs(self.args.rollout_save_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        filename_parts = [
            "rollout",
            self._slugify(policy_name) if policy_name else None,
            self._slugify(instruction),
            self._slugify(success_label) if success_label else None,
            timestamp,
        ]
        save_filename = "_".join(part for part in filename_parts if part) + ".h5"
        save_path = os.path.join(self.args.rollout_save_dir, save_filename)

        # Subsample rate
        rate = self.args.rollout_subsample_rate

        # Convert lists to numpy arrays with subsampling
        cartesian_positions = np.stack(rollout_data["cartesian_positions"][::rate])  # (N, 6)
        gripper_positions = np.array(rollout_data["gripper_positions"][::rate])  # (N,)
        actions = np.stack(rollout_data["actions"][::rate])  # (N, 7)

        with h5py.File(save_path, "w") as f:
            # Observation group
            obs_group = f.create_group("observation")

            # Robot state
            robot_state = obs_group.create_group("robot_state")
            robot_state.create_dataset("cartesian_position", data=cartesian_positions)
            robot_state.create_dataset("gripper_position", data=gripper_positions)

            # Images (subsampled)
            image_group = obs_group.create_group("image")
            if rollout_data["side_images"]:
                side_images = np.stack(rollout_data["side_images"][::rate])  # (N, H, W, C)
                image_group.create_dataset("side", data=side_images, compression="gzip")
            if rollout_data.get("wrist_images"):
                wrist_images = np.stack(rollout_data["wrist_images"][::rate])  # (N, H, W, C)
                image_group.create_dataset("wrist", data=wrist_images, compression="gzip")

            # Action group
            action_group = f.create_group("action")
            action_group.create_dataset("cartesian_position", data=actions)

            # Metadata
            f.attrs["instruction"] = instruction
            f.attrs["subsample_rate"] = rate
            if policy_name:
                f.attrs["policy_name"] = policy_name
            if success_label:
                f.attrs["success"] = success_label == "success"

        print(f"Rollout saved to {save_path} (subsampled by {rate}x, {len(actions)} frames)")

    def _rollout_once(self, instruction, fetch_action_chunk, refresh_horizon, print_action=False):
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        chunk_length = refresh_horizon
        video = []
        wrist_video = []

        # Rollout data collection for h5 saving
        rollout_data = {
            "cartesian_positions": [],
            "gripper_positions": [],
            "actions": [],
            "side_images": [],
            "wrist_images": [],
        }

        for t in range(self.args.max_timesteps):
            start_time = time.time()
            try:
                curr_obs = self._extract_observation(self.env.get_observation(), save_to_disk=t == 0)
                self._record_frames(curr_obs, video, wrist_video)

                # Collect observation data for h5 saving
                if self.args.save_rollout:
                    rollout_data["cartesian_positions"].append(
                        np.asarray(curr_obs["cartesian_position"], dtype=np.float64)
                    )
                    rollout_data["gripper_positions"].append(float(curr_obs["gripper_position"]))
                    # Collect images
                    side_frame = curr_obs.get(self.side_image_name)
                    if side_frame is None and self.args.external_camera is not None:
                        side_frame = curr_obs.get(f"{self.args.external_camera}_image")
                    if side_frame is not None:
                        img = side_frame[0] if len(side_frame.shape) == 4 else side_frame
                        rollout_data["side_images"].append(img.copy())
                    wrist_frame = curr_obs.get("wrist_image")
                    if wrist_frame is not None:
                        img = wrist_frame[0] if len(wrist_frame.shape) == 4 else wrist_frame
                        rollout_data["wrist_images"].append(img.copy())

                if pred_action_chunk is None or actions_from_chunk_completed >= chunk_length:
                    actions_from_chunk_completed = 0
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = fetch_action_chunk(curr_obs)
                        chunk_length = min(refresh_horizon, len(pred_action_chunk))
                action = pred_action_chunk[actions_from_chunk_completed]
                action = self.binarize_gripper(action)
                actions_from_chunk_completed += 1
                if print_action:
                    print(action)

                # Collect action data for h5 saving (action sent to env.step)
                if self.args.save_rollout:
                    rollout_data["actions"].append(np.asarray(action, dtype=np.float64))
                self.env.step(action)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        policy_name = None
        success_label = None
        save_video = input("Save video? (enter y or n) ")
        if "y" in save_video.lower():
            policy_name = input("Policy name for this rollout? ").strip()
            success_answer = input("Was the rollout successful? (enter y or n) ")
            success_label = "success" if "y" in success_answer.lower() else "fail"
            self._concat_and_save_video(
                instruction, video, wrist_video, policy_name=policy_name, success_label=success_label
            )

        # Save rollout to h5 if enabled
        if self.args.save_rollout:
            if policy_name is None:
                policy_name = input("Policy name for h5 rollout? ").strip()
            if success_label is None:
                success_answer = input("Was the rollout successful? (enter y or n) ")
                success_label = "success" if "y" in success_answer.lower() else "fail"
            self._save_rollout_to_h5(rollout_data, instruction, policy_name=policy_name, success_label=success_label)

    def _reset_until_confirmed(self):
        while True:
            self.env.reset()
            answer = input("Correctly reset (enter y or n)? ")
            if "n" not in answer.lower():
                break

    def _run_sessions(self, make_fetcher, refresh_horizon, print_action=False):
        while True:
            instruction = input("Enter instruction: ")
            print("Running rollout... press Ctrl+C to stop early.")
            self._rollout_once(instruction, make_fetcher(instruction), refresh_horizon, print_action=print_action)
            answer = input("Do one more eval? (enter y or n) ")
            if "n" in answer.lower():
                if hasattr(self.env, "close"):
                    self.env.close()
                    os._exit(0)
                break
            self._reset_until_confirmed()

    def run(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)

        def make_fetcher(instruction):
            return lambda curr_obs: self.get_action_from_response(
                policy_client.infer(self.obs_to_request(curr_obs, instruction)),
                curr_obs,
            )

        self._run_sessions(make_fetcher, refresh_horizon=self.CHUNK_STEPS)

    def run_upstream(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)

        def make_fetcher(instruction):
            return lambda curr_obs: policy_client.infer(self.obs_to_request(curr_obs, instruction))["actions"]

        self._run_sessions(make_fetcher, refresh_horizon=self.args.open_loop_horizon, print_action=True)
