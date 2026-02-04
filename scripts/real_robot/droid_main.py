# ruff: noqa
import numpy as np
from droid.robot_env import RobotEnv
import tyro
import sys

sys.path.append(".")
from shared import BaseEvalRunner, Args

from helpers import binarize_gripper_actions_np, invert_gripper_actions_np


class DroidEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = f"{self.args.external_camera}_image"

    def resolve_camera_keys(self, image_observations):
        side_key = self.find_camera_key(image_observations, self.args.left_camera_id, require_left_stream=True)
        wrist_key = self.find_camera_key(image_observations, self.args.wrist_camera_id, require_left_stream=True)
        return side_key, wrist_key

    def process_gripper_observation(self, gripper_position: np.ndarray) -> np.ndarray:
        return binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)


class DroidUpstreamEvalRunner(DroidEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        return RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
        )


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.run_upstream:
        eval_runner = DroidUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        print("Running in base frame")
        eval_runner = DroidEvalRunner(args)
        eval_runner.run()
