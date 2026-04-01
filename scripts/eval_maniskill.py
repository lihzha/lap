#!/usr/bin/env python3
"""Evaluate LAP (Language-Action Pre-training) policies on ManiSkill PlugCharger-v1.

Run from the language-action-pretraining project root:
    uv run scripts/eval_maniskill.py
    uv run scripts/eval_maniskill.py --checkpoint_dir checkpoints/lap --num_episodes 50
    uv run scripts/eval_maniskill.py --policy_type ar --save_video true
"""

import collections
import dataclasses
import json
import logging
from pathlib import Path
import sys

import gymnasium as gym
import h5py
import imageio
import numpy as np
from openpi_client import image_tools
from scipy.spatial.transform import Rotation
import torch
import tyro

# Register ManiSkill environments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ManiSkill"))
import mani_skill.envs  # noqa: F401

import lap.policies.policy_config_adapter as _policy_config
from lap.training import config as _config

GRIPPER_MIN = 0.0135
GRIPPER_MAX = 0.0400

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    # Policy
    config: str = "lap"
    """Training config name (e.g. 'lap', 'lap_libero')."""

    checkpoint_dir: str = "checkpoints/lap"
    """Path to checkpoint directory (contains params/ and assets/)."""

    policy_type: str = "flow"
    """'flow' uses the action expert; 'ar' uses autoregressive language-action sampling."""

    default_prompt: str = "Plug the charger."
    """Task instruction passed to the policy."""

    # Demo trajectories
    demo_json: str = (
        "/home/irom-lab/.maniskill/demos/PlugCharger-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.json"
    )
    """Path to the trajectory JSON listing valid episodes."""

    demo_h5: str = (
        "/home/irom-lab/.maniskill/demos/PlugCharger-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    )
    """Path to the HDF5 file with recorded env states."""

    # Evaluation
    num_episodes: int = 50
    """Number of episodes to evaluate."""

    replan_steps: int = 8
    """Execute this many actions from a predicted chunk before re-querying the policy."""

    max_steps: int = 300
    """Maximum environment steps per episode (env default is 200)."""

    # Environment
    sim_backend: str = "physx_cpu"
    """Simulation backend: 'physx_cpu' or 'physx_cuda'."""

    seed: int = 0
    """Base random seed (episode i uses seed + i)."""

    # Output
    save_video: bool = True
    """Save per-episode rollout videos to video_dir."""

    video_dir: str = "data/maniskill_eval_videos"
    """Directory for saved rollout videos."""


# ---------------------------------------------------------------------------
# Demo trajectory helpers
# ---------------------------------------------------------------------------


def _load_demo_episodes(json_path: str) -> list[dict]:
    """Return the list of episode metadata dicts from the trajectory JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data["episodes"]


def _set_initial_state_from_h5(env: gym.Env, h5_file: h5py.File, episode_id: int) -> None:
    """Override the environment's state with the first frame of traj_{episode_id}."""
    traj = h5_file[f"traj_{episode_id}"]
    env_states = traj["env_states"]

    state_dict: dict = {"actors": {}, "articulations": {}}
    for actor_name, data in env_states["actors"].items():
        state_dict["actors"][actor_name] = torch.tensor(data[0:1], dtype=torch.float32)
    for art_name, data in env_states["articulations"].items():
        state_dict["articulations"][art_name] = torch.tensor(data[0:1], dtype=torch.float32)

    env.unwrapped.set_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------

_IMAGE_SIZE = 224
_CONTROL_MODE = "pd_joint_pos"


def _build_policy_obs(obs: dict, prompt: str) -> dict:
    """Convert a ManiSkill observation (batch size 1) to the LAP policy input format.

    Expected policy input keys:
        base_0_rgb      (224, 224, 3) uint8 – base/overhead camera
        left_wrist_0_rgb (224, 224, 3) uint8 – wrist camera
        state           (10,) float32 – [eef_pos(3), rot6d(6), gripper(1)]
        prompt          str
    """
    # --- Images -----------------------------------------------------------
    # ManiSkill sensor_data rgb: torch.Tensor (num_envs, H, W, 3) uint8, possibly on CUDA
    base_rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()  # (H, W, 3)
    wrist_rgb = obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy()  # (H, W, 3)

    base_rgb = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_rgb, _IMAGE_SIZE, _IMAGE_SIZE))
    wrist_rgb = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_rgb, _IMAGE_SIZE, _IMAGE_SIZE))

    # --- Robot state ------------------------------------------------------
    # tcp_pose raw_pose layout: [x, y, z, qw, qx, qy, qz]
    tcp_raw = obs["extra"]["tcp_pose"][0].cpu().numpy().astype(np.float32)  # (7,)
    eef_pos = tcp_raw[:3]
    # Convert wxyz → xyzw for scipy (scalar-last convention)
    quat_xyzw = np.array([tcp_raw[4], tcp_raw[5], tcp_raw[6], tcp_raw[3]], dtype=np.float64)
    rot_matrix = Rotation.from_quat(quat_xyzw).as_matrix()  # (3, 3)
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0).astype(np.float32)  # (6,)

    # Gripper: panda finger joint position in [0, 0.04]; normalize to [0, 1]
    gripper_norm = float(obs["agent"]["qpos"][0, -2].cpu() - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)
    gripper_norm = np.clip(gripper_norm, 0.0, 1.0)

    state = np.concatenate([eef_pos, rot6d, [gripper_norm]], dtype=np.float32)  # (10,)

    return {
        "observation": {
            "base_0_rgb": base_rgb,
            "left_wrist_0_rgb": wrist_rgb,
            "state": state,
        },
        "prompt": prompt,
    }


def _convert_action(policy_action: np.ndarray) -> np.ndarray:
    """Map a single LAP policy action to ManiSkill pd_ee_delta_pose action.

    LAP output:  [dx, dy, dz, droll, dpitch, dyaw, gripper_01]  (7,)
    ManiSkill:   [dx, dy, dz, droll, dpitch, dyaw, gripper_pos] (7,)
        - Position delta (m), clipped to the controller range [-0.1, 0.1].
        - Rotation delta as euler XYZ (rad), clipped to [-0.1, 0.1].
        - Gripper target finger position in [-0.01, 0.04]:
            policy output 1.0 (open)  → 0.04
            policy output 0.0 (closed) → 0.0
    """
    action = policy_action.copy().astype(np.float32)

    # Clip to controller limits
    # action[:3] = np.clip(action[:3], -0.1, 0.1)
    # action[3:6] = np.clip(action[3:6], -0.1, 0.1)

    # Map gripper from normalised [0, 1] to finger-joint target [0, 0.04]
    action[-1] = np.clip(float(action[-1]) * 2 - 1, -1.0, 1.0)

    return action


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    logging.info("Loading config '%s' from %s …", args.config, args.checkpoint_dir)
    train_config = _config.get_config(args.config)
    # Disable training-only flag before inference
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, stop_action_to_vlm_grad=False),
    )

    if args.policy_type == "ar":
        policy = _policy_config.create_trained_policy_ar(
            train_config, args.checkpoint_dir, default_prompt=args.default_prompt
        )
    elif args.policy_type == "flow":
        policy = _policy_config.create_trained_policy(
            train_config, args.checkpoint_dir, default_prompt=args.default_prompt
        )
    else:
        raise ValueError(f"Unknown policy_type '{args.policy_type}'. Choose 'flow' or 'ar'.")
    logging.info("Policy loaded.")

    # ------------------------------------------------------------------
    # Load demo trajectories
    # ------------------------------------------------------------------
    demo_episodes = _load_demo_episodes(args.demo_json)
    logging.info("Loaded %d demo episodes from %s", len(demo_episodes), args.demo_json)
    demo_h5 = h5py.File(args.demo_h5, "r")

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    env = gym.make(
        "PlugCharger-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode=_CONTROL_MODE,
        sim_backend=args.sim_backend,
        reconfiguration_freq=1,  # randomise scene every reset
        render_mode=None,  # headless — no viewer window
    )
    logging.info("Environment created (backend=%s, control_mode=%s).", args.sim_backend, _CONTROL_MODE)

    if args.save_video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    sampled_episodes = rng.choice(demo_episodes, size=args.num_episodes, replace=args.num_episodes > len(demo_episodes))
    total_episodes = 0
    total_successes = 0

    for ep, demo_ep in enumerate(sampled_episodes):
        episode_id = demo_ep["episode_id"]
        episode_seed = demo_ep["reset_kwargs"]["seed"]
        obs, _ = env.reset(seed=episode_seed)
        # _set_initial_state_from_h5(env, demo_h5, episode_id)
        obs = env.unwrapped.get_obs()
        logging.info("Episode %d: using demo episode_id=%d seed=%d", ep + 1, episode_id, episode_seed)

        action_plan: collections.deque = collections.deque()
        success = False
        frames = []

        for step in range(args.max_steps):
            if args.save_video:
                base_frame = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
                wrist_frame = obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy()
                base_frame = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(base_frame, _IMAGE_SIZE, _IMAGE_SIZE)
                )
                wrist_frame = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_frame, _IMAGE_SIZE, _IMAGE_SIZE)
                )
                frames.append(np.concatenate([base_frame, wrist_frame], axis=1))

            # Query policy when the current action chunk is exhausted
            if not action_plan:
                policy_obs = _build_policy_obs(obs, args.default_prompt)
                result = policy.infer(policy_obs)
                actions = result["actions"]  # (action_horizon, 7)
                chunk = actions[: args.replan_steps]
                action_plan.extend(_convert_action(actions[i]) for i in range(len(chunk)))

            action = action_plan.popleft()
            # ManiSkill vectorised env expects shape (num_envs, action_dim)
            obs, _reward, terminated, truncated, info = env.step(np.expand_dims(action, axis=0))

            # Success is reported per-step by evaluate()
            if "success" in info:
                success = bool(np.asarray(info["success"]).ravel()[0])
                if success:
                    break

            if terminated.any() or truncated.any():
                break

        total_episodes += 1
        total_successes += int(success)
        sr = total_successes / total_episodes

        logging.info(
            "Episode %d/%d: %s  |  running SR %d/%d = %.1f%%",
            ep + 1,
            args.num_episodes,
            "SUCCESS" if success else "FAILURE",
            total_successes,
            total_episodes,
            sr * 100,
        )

        if args.save_video and frames:
            try:
                suffix = "success" if success else "failure"
                out_path = f"{args.video_dir}/episode_{ep:03d}_{suffix}.mp4"
                imageio.mimwrite(out_path, frames, fps=10)
                logging.info("Video saved to %s", out_path)
            except Exception as exc:
                logging.warning("Could not save video: %s", exc)

    env.close()
    demo_h5.close()

    final_sr = total_successes / total_episodes if total_episodes else 0.0
    logging.info(
        "=== Final results: %d/%d episodes succeeded (%.1f%% success rate) ===",
        total_successes,
        total_episodes,
        final_sr * 100,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
