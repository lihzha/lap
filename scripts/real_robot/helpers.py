import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import contextlib
import signal
import dataclasses

def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
    rot_matrix = R.from_euler("xyz", euler_angles, degrees=False).as_matrix()
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
    return rot6d

def binarize_gripper_actions_np(actions: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Convert continuous gripper actions to binary (0 or 1) using backward propagation logic.
    """
    actions = actions.astype(np.float32)
    n = actions.shape[0]
    new_actions = np.zeros_like(actions)

    open_mask = actions > threshold
    closed_mask = actions < (1 - threshold)
    in_between_mask = ~(open_mask | closed_mask)

    carry = actions[-1] > threshold  # carry as boolean (True=open)
    
    for i in reversed(range(n)):
        if not in_between_mask[i]:
            carry = open_mask[i]
        new_actions[i] = float(carry)

    return new_actions

def invert_gripper_actions_np(actions: np.ndarray) -> np.ndarray:
    """Invert gripper binary actions: 1 → 0, 0 → 1."""
    return 1.0 - actions

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

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

@dataclasses.dataclass(frozen=True)
class ActionChunkPostProcessor:
    """Convert policy responses into executable robot action chunks."""

    chunk_steps: int = 8
    use_quaternions: bool = False

    def process_response(
        self,
        response: dict,
        *,
        curr_pos: np.ndarray,
        curr_rpy: np.ndarray,
        gripper_postprocess_fn,
        orientation_interpolator_fn,
    ) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=float)
        if actions.ndim == 1:
            return self._single_action_to_chunk(
                actions,
                curr_pos=curr_pos,
                curr_rpy=curr_rpy,
                gripper_postprocess_fn=gripper_postprocess_fn,
                orientation_interpolator_fn=orientation_interpolator_fn,
            )
        return self._chunk_actions(
            actions,
            curr_pos=curr_pos,
            curr_rpy=curr_rpy,
            gripper_postprocess_fn=gripper_postprocess_fn,
        )

    def _single_action_to_chunk(
        self,
        action: np.ndarray,
        *,
        curr_pos: np.ndarray,
        curr_rpy: np.ndarray,
        gripper_postprocess_fn,
        orientation_interpolator_fn,
    ) -> np.ndarray:
        grip_action = float(np.asarray(gripper_postprocess_fn(action)).reshape(-1)[0])
        positions = np.linspace(curr_pos, curr_pos + action[:3], self.chunk_steps, endpoint=True)
        rpy_arr = orientation_interpolator_fn(curr=curr_rpy, delta=action[3:6], steps=self.chunk_steps)
        grip_vals = np.full((self.chunk_steps, 1), grip_action)

        if self.use_quaternions:
            quat_arr = R.from_euler("xyz", rpy_arr, degrees=False).as_quat()
            return np.concatenate([positions, quat_arr, grip_vals], axis=1)
        return np.concatenate([positions, rpy_arr, grip_vals], axis=1)

    def _chunk_actions(
        self,
        pred_action_chunk: np.ndarray,
        *,
        curr_pos: np.ndarray,
        curr_rpy: np.ndarray,
        gripper_postprocess_fn,
    ) -> np.ndarray:
        pred_action_chunk = pred_action_chunk.copy()
        if pred_action_chunk.shape[-1] > 7:
            return pred_action_chunk

        pred_action_chunk[:, :3] += curr_pos
        pred_action_chunk[:, 3:6] = add_euler(curr=curr_rpy, delta=pred_action_chunk[:, 3:6])

        pred_action_chunk[:, -1] = gripper_postprocess_fn(pred_action_chunk)
        if not self.use_quaternions:
            return pred_action_chunk

        quat_arr = R.from_euler("xyz", pred_action_chunk[:, 3:6], degrees=False).as_quat()
        return np.concatenate([pred_action_chunk[:, :3], quat_arr, pred_action_chunk[:, 6:7]], axis=1)


def add_euler(curr: np.ndarray, delta: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """Add Euler-angle deltas to a current Euler rotation.

    Args:
        curr: (3,) array for current Euler angles [roll, pitch, yaw].
        delta: (..., 3) array of Euler deltas.
        seq: Rotation sequence (default: "xyz", extrinsic).

    Returns:
        (..., 3) array of updated Euler angles.
    """
    curr = np.asarray(curr, dtype=float)
    delta = np.asarray(delta, dtype=float)
    r_curr = R.from_euler(seq, curr)
    r_delta = R.from_euler(seq, delta)
    return (r_curr * r_delta).as_euler(seq)