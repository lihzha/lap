"""PixmoPoint dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.registry import register_dataset
from lap.datasets.vqa.vqa_base import BaseVQADataset

# Maximum number of points to include in the answer
MAX_POINTS = 20

# Prompts for PixmoPoint dataset - split into prefix and suffix for label insertion
# Contains both general prompts and robot-specific manipulation prompts
PIXMO_POINT_PROMPT_PARTS = [
    # ---- GENERAL POINT PROMPTS (kept from original) ----
    ("How many ", " are in the image? Point them out."),
    ("Point out all the ", " in this image."),
    ("Where are the ", " in the image? Point to each one."),
    ("Locate all ", " in the image and point them out."),
    ("Point to ", ". Please say 'There are none.' if it is not in the image."),
    ("Point to all occurrences of ", "."),
    ("Point to any ", " in the image."),
    ("Point: Where are the ", "?"),
    ("Show me where the ", " are."),
    ("If there are any ", " in the image, show me where they are."),
    ("Where are the ", "?"),
    ("Generate a list of points showing where the ", " are."),
    ("Find the ", "."),
    ("Locate all ", "."),
    ("Locate the ", "."),
    ("Object: ", ". Instruction: Point to the object."),
    ("find ", "."),
    ("Point to every ", "."),
    ("Find any ", "."),
    ("Point to a ", "."),
    ("Look for ", " in the image and show me where they are."),
    ("Help me find an object in the image by pointing to it. Object: ", "."),
    ("I am looking for ", ", where can it be found in the image?"),
    ("Can you see any ", " in the image? Point to them."),
    ("Point out each ", " in the image."),
    # ---- ROBOT-SPECIFIC MANIPULATION PROMPTS ----
    ("Show me where the robot should move its end-effector to reach the ", " in the image."),
    ("Point to where the robot should position its gripper to grasp the ", "."),
    ("Locate the point where the robot should align its end-effector with the ", " in the image."),
    ("Mark the location the robot should target with its gripper to reach the ", "."),
    ("Identify the spot the robot should move its arm toward to approach the ", "."),
    ("Point to the region the robot should aim its end-effector at to interact with the ", "."),
    ("Show me the point where the robot would position its gripper to approach the ", " in the image."),
    ("Indicate where the robot should move its arm to reach the ", "."),
    ("Point to the location the robot should target to interact with the ", "."),
    ("Highlight the point the robot should move toward to grasp the ", "."),
    ("Identify where the robot should position its wrist relative to the ", "."),
    ("Point out the spot the robot would navigate its arm to in order to reach the ", "."),
    ("Locate where the robot would need to move its end-effector to get closer to the ", " in the image."),
    ("Point to the position the robot should move its gripper toward to access the ", "."),
    ("Show the point the robot should aim its arm toward when approaching the ", "."),
    ("Indicate the exact point a robot should target with its gripper when reaching for the ", "."),
    ("Point to where the robot should aim its wrist to reach the ", "."),
    ("Mark the precise point where the robot should position its end-effector to approach the ", "."),
    ("Identify the point where the robot would place its gripper to interact with the ", "."),
    ("Show the location the robot should move its arm to reach the ", "."),
    ("Locate the target point the robot should align its manipulator with to access the ", "."),
    ("Point out the position the robot would need to occupy with its wrist to manipulate the ", "."),
    ("Point to the region that represents the robot's goal location for reaching the ", "."),
    ("Find the point in the image that the robot should move its end-effector toward to reach the ", "."),
    ("Mark the destination point a robot should target with its gripper to successfully approach the ", "."),
]


@register_dataset(name="pixmo_point", is_vqa=True)
class PixmoPoint(BaseVQADataset):
    """PixmoPoint dataset for vision-language training with point annotations.

    This dataset loads images with point annotations and formats them to be
    compatible with the robot dataset structure. Points are converted to text
    format with normalized coordinates.
    """

    def __init__(self, *args, max_points: int = MAX_POINTS, scale: float = 1.0, **kwargs):
        """Initialize PixmoPoint dataset.

        Args:
            max_points: Maximum number of points to include (extras are dropped).
            scale: Scale factor for normalizing coordinates (default 100).
            *args, **kwargs: Passed to base class.
        """
        self.max_points = max_points
        self.scale = scale
        super().__init__(*args, **kwargs)

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for PixmoPoint."""

        return tfds.builder("pixmo_point", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "pixmo_point"

    def get_num_transitions(self) -> int:
        """Return approximate number of PixmoPoint samples."""
        return 1702160

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from PixmoPoint image SHA256, label, and count.

        Since the same image can have multiple different annotations (different labels
        or different sets of points), we need to include annotation-specific information
        to ensure uniqueness.
        """
        image_sha = example["image_sha256"]
        label = example["label"]
        count = example["count"]
        # Create a hash of the combination to ensure uniqueness
        combined = tf.strings.join([image_sha, "_", label, "_", tf.strings.as_string(count)])
        combined_hash = tf.strings.to_hash_bucket_fast(combined, 2147483647)
        combined_hash_str = tf.strings.as_string(combined_hash)
        return tf.strings.join(["pixmo_point_", image_sha, "_", combined_hash_str])

    def points_to_text(
        self,
        points: tf.Tensor,
        # count_text: tf.Tensor,
    ) -> tf.Tensor:
        """Convert points to formatted text representation using paligemma loc tokens.

        Args:
            points: Tensor of shape [N, 2] with x, y coordinates (already normalized to 0-100).

        Returns:
            Formatted string with points as paligemma loc tokens like "<loc0252><loc0233>".
        """
        # Sort points by x*10000 + y for consistent ordering
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        sort_keys = x_coords * 10000.0 + y_coords
        sorted_indices = tf.argsort(sort_keys)
        sorted_points = tf.gather(points, sorted_indices)

        # Extract sorted x and y coordinates
        x_coords = sorted_points[:, 0]
        y_coords = sorted_points[:, 1]

        # Convert to paligemma loc token indices (0-1023)
        N = 1024
        x_indices = tf.cast(tf.round(x_coords / 100.0 * (N - 1)), tf.int32)
        y_indices = tf.cast(tf.round(y_coords / 100.0 * (N - 1)), tf.int32)

        # Format as loc tokens: <locYYYY><locXXXX> for each point
        y_tokens = tf.strings.join(["<loc", tf.strings.as_string(y_indices, width=4, fill="0"), ">"])
        x_tokens = tf.strings.join(["<loc", tf.strings.as_string(x_indices, width=4, fill="0"), ">"])

        # Interleave y and x tokens (y comes first for each point)
        yx_pairs = tf.reshape(tf.stack([y_tokens, x_tokens], axis=1), [-1])

        # Join all tokens together without separator
        return tf.strings.reduce_join(yx_pairs, separator="")

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from PixmoPoint example.

        Returns:
            (prompt, caption) where prompt asks about the label and caption
            contains the formatted points.
        """
        label = example["label"]
        points_data = example["points"]
        count = example["count"]

        # Extract x, y coordinates and stack them
        x_coords = points_data["x"]
        y_coords = points_data["y"]
        points = tf.stack([x_coords, y_coords], axis=1)  # Shape: [N, 2]

        # Normalize to 0-100 scale and round to 1 decimal place
        points = points * (100.0 / self.scale)
        points = tf.round(points * 10.0) / 10.0  # Round to 1 decimal

        # Limit to max_points
        num_points = tf.shape(points)[0]
        num_points_capped = tf.minimum(num_points, self.max_points)
        points = points[:num_points_capped]

        # Generate deterministic seed from image SHA256
        image_sha = example["image_sha256"]
        sha_hash = tf.strings.to_hash_bucket_fast(image_sha, 2147483647)
        sha_hash_int = tf.cast(sha_hash, tf.int32)

        # Randomly select a prompt template using tf.switch_case
        num_prompts = len(PIXMO_POINT_PROMPT_PARTS)
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, sha_hash_int], minval=0, maxval=num_prompts, dtype=tf.int32
        )

        # Create branches for each prompt template
        def make_prompt_fn(idx):
            prefix, suffix = PIXMO_POINT_PROMPT_PARTS[idx]

            def fn():
                return tf.strings.join([prefix, label, suffix])

            return fn

        prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
        prompt = tf.switch_case(prompt_idx, branch_fns=prompt_branches)

        # Format points as answer/caption
        # count_text = tf.strings.join([tf.string(count), " ", label])
        caption = self.points_to_text(points)

        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode PixmoPoint image to JPEG bytes."""
        image = example["image"]
        # Check if image is already encoded (has dtype string)
        if image.dtype == tf.string:
            # Already encoded, return as-is
            return image
        # Not encoded, encode to JPEG
        return tf.io.encode_jpeg(image, quality=95)
