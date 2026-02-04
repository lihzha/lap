import collections
import dataclasses
import datetime
import enum
import json
import logging
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class PolicyType(str, enum.Enum):
    """Supported policy serving modes for LIBERO eval."""

    LAP = "LAP"
    LAP_AR = "LAP_AR"


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    policy_type: PolicyType = PolicyType.LAP

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    control_mode: str = "OSC_POSE"  # Controller type. Options: OSC_POSE, IK_POSE, OSC_POSITION, JOINT_POSITION, etc.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "data/libero/results"  # Path to save evaluation results

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize results tracking
    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_suite": args.task_suite_name,
            "policy_type": args.policy_type.value,
            "control_mode": args.control_mode,
            "seed": args.seed,
            "num_trials_per_task": args.num_trials_per_task,
            "replan_steps": args.replan_steps,
        },
        "episodes": [],
        "per_task_results": [],
        "summary": {},
    }

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # max_steps = 50

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_mode)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            wrist_replay_images = []
            episode_start_time = datetime.datetime.now()

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img, wrist_img = get_images_from_obs(obs, args.resize_size)

                    if not action_plan:
                        # Query model to get action
                        request = obs_to_request(obs, args.policy_type, img, wrist_img, task_description)
                        response = client.infer(request)
                        single_action_or_chunk = np.asarray(response["actions"], dtype=np.float32)
                        if single_action_or_chunk.ndim == 1:
                            assert args.policy_type == PolicyType.LAP_AR
                            action_chunk = get_action_from_response(
                                args.replan_steps, response, request["observation"]["state"]
                            )
                        else:
                            action_chunk = single_action_or_chunk
                            action_chunk = invert_and_scale_gripper(single_action_or_chunk)
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # Save preprocessed image for replay video
                    # Draw reasoning on image if available
                    # if "reasoning" in response and response["reasoning"] is not None:
                    #     img_with_text = _draw_text_on_image(img, response["reasoning"])
                    #     replay_images.append(img_with_text)
                    # else:
                    replay_images.append(img)
                    wrist_replay_images.append(wrist_img)

                    action = action_plan.popleft()
                    # action = np.concatenate([action[:3], _euler2axisangle(action[3:6]), action[6:]])

                    # Execute action in environment
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Calculate episode duration
            episode_duration = (datetime.datetime.now() - episode_start_time).total_seconds()

            # Record episode results
            episode_result = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_id": episode_idx,
                "global_episode_id": total_episodes - 1,
                "success": bool(done),
                "num_steps": t - args.num_steps_wait,
                "total_steps_with_wait": t,
                "duration_seconds": episode_duration,
            }
            all_results["episodes"].append(episode_result)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_wrist_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in wrist_replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Record per-task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_result = {
            "task_id": task_id,
            "task_description": task_description,
            "num_episodes": task_episodes,
            "num_successes": task_successes,
            "success_rate": task_success_rate,
        }
        all_results["per_task_results"].append(task_result)

        # Log final results
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Calculate and save final summary
    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    all_results["summary"] = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "num_tasks": num_tasks_in_suite,
    }

    # Save results to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{args.task_suite_name}_{args.policy_type.value}_{timestamp}.json"
    results_path = pathlib.Path(args.results_out_path) / results_filename
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Total success rate: {overall_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Results saved to: {results_path}")


def _get_libero_env(task, resolution, seed, controller="OSC_POSE"):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "controller": controller,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_images_from_obs(obs, resize_size):
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][:, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    return img, wrist_img


def obs_to_request(obs, policy_type: PolicyType, img, wrist_img, task_description: str):
    # Prepare observations dict
    assert policy_type in (PolicyType.LAP_AR, PolicyType.LAP_AR), f"Unsupported policy type: {policy_type}"
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_rot6d = _quat2rot6d(obs["robot0_eef_quat"]).astype(np.float32, copy=False)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper_state = np.clip(gripper_qpos[-2:-1] / 0.04, 0, 1)  # normalize to [0, 1]
    state = np.concatenate((eef_pos, eef_rot6d, gripper_state)).astype(np.float32, copy=False)
    return {
        "observation": {
            "base_0_rgb": img,
            "left_wrist_0_rgb": wrist_img,
            "state": state,
        },
        "prompt": str(task_description),
    }


def invert_and_scale_gripper(action_chunk):
    action_chunk[:, -1:] = 1 - 2 * action_chunk[:, -1:]
    action_chunk[:, -1:] = np.sign(action_chunk[:, -1:])
    return action_chunk


def _quat2rot6d(quat):
    "Convert quaternion to 6D rotation representation."
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,), ordered as [x, y, z, w]")
    rot_matrix = R.from_quat(q).as_matrix()
    # rot6d = rot_matrix[:, :2].flatten()
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
    return rot6d


def get_action_from_response(replan_steps, response, state):
    curr_pos = np.asarray(state[:3], dtype=float)
    curr_rpy = np.asarray(state[3:6], dtype=float)

    action = np.asarray(response["actions"])
    grip_action = action[-1]
    print(grip_action)

    # Linearly interpolate to replan_steps actions
    positions = np.linspace(curr_pos, curr_pos + action[:3], replan_steps, endpoint=True)
    rpy_arr = interpolate_rpy(curr=curr_rpy, delta=action[3:6], steps=replan_steps)
    grip_vals = np.full((replan_steps, 1), grip_action)
    # grip_vals[: self.replan_steps // 2] = 1 - curr_obs["gripper_position"]
    return np.concatenate([positions, rpy_arr, grip_vals], axis=1)


def interpolate_rpy(curr, delta, steps):
    """Interpolate roll-pitch-yaw angles using quaternion SLERP.

    This function uses spherical linear interpo lation (SLERP) on quaternions
    to provide smooth rotation interpolation, avoiding gimbal lock and
    discontinuities that occur with naive linear interpolation of Euler angles.

    Args:
        curr: Current RPY angles as array of shape (3,) in radians
        delta: Change in RPY angles as array of shape (3,) or (n, 3) in radians
        steps: Number of interpolation steps

    Returns:
        Array of shape (steps, 3) with interpolated RPY values in radians
    """
    curr = np.asarray(curr, dtype=float)
    delta = np.asarray(delta, dtype=float)

    # Handle both 1D and 2D delta inputs
    if delta.ndim == 1:
        # Single delta vector - interpolate from curr to curr + delta
        target_rpy = curr + delta
    else:
        # Multiple deltas - use the first one
        target_rpy = curr + delta[0] if len(delta) > 0 else curr

    # Convert current and target RPY to rotation objects
    # RPY convention: rotate around x (roll), then y (pitch), then z (yaw)
    rot_curr = R.from_euler("xyz", curr, degrees=False)
    rot_target = R.from_euler("xyz", target_rpy, degrees=False)

    # Create SLERP interpolator
    key_times = np.array([0, 1])
    key_rots = R.concatenate([rot_curr, rot_target])
    slerp = Slerp(key_times, key_rots)

    # Generate interpolation times
    interp_times = np.linspace(0, 1, steps, endpoint=True)

    # Perform SLERP interpolation
    interpolated_rots = slerp(interp_times)

    # Convert back to RPY
    rpy_arr = interpolated_rots.as_euler("xyz", degrees=False)

    return rpy_arr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
