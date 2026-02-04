# ruff: noqa
import numpy as np
import tyro
import sys

sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np


class FrankaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = "right_image"

    def use_quaternion_actions(self) -> bool:
        return True

    def resolve_camera_keys(self, image_observations):
        side_key = self.find_camera_key(image_observations, self.args.left_camera_id, require_left_stream=True)
        wrist_key = self.find_camera_key(image_observations, self.args.wrist_camera_id, require_left_stream=False)
        return side_key, wrist_key

    def process_gripper_observation(self, gripper_position: np.ndarray) -> np.ndarray:
        return binarize_gripper_actions_np(gripper_position)


class FrankaUpstreamEvalRunner(FrankaEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        from droid.robot_env import RobotEnv

        return RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
        )

    def wrist_camera_rotate_180(self) -> bool:
        return False

    def use_rot6d_state(self) -> bool:
        return False

    def process_gripper_observation(self, gripper_position: np.ndarray) -> np.ndarray:
        return 1 - binarize_gripper_actions_np(gripper_position)


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.run_upstream:
        eval_runner = FrankaUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        print("Running in base frame")
        eval_runner = FrankaEvalRunner(args)
        eval_runner.run()
