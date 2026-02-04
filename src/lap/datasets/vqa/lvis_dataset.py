"""LVIS dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.registry import register_dataset
from lap.datasets.vqa.bbox import DIRECTION_PROMPT_PARTS
from lap.datasets.vqa.bbox import ROBOT_BBOX_PROMPT_PARTS_OXE
from lap.datasets.vqa.bbox import bbox_to_text_tf
from lap.datasets.vqa.bbox import direction_from_bbox_tf
from lap.datasets.vqa.bbox import sample_prompt_tf
from lap.datasets.vqa.vqa_base import BaseVQADataset


@register_dataset(name="lvis", is_vqa=True)
class Lvis(BaseVQADataset):
    """LVIS dataset for vision-language training.

    This dataset loads LVIS images with object annotations and formats them to be
    compatible with the robot dataset structure. Each image contains multiple object
    annotations with category labels.
    """

    def __init__(self, *args, directional: bool = False, direction_slope: float = 2.0, **kwargs):
        self.directional = directional
        self.direction_slope = direction_slope
        # Get direction_prob from config before calling super().__init__()
        # because super().__init__() calls apply_vqa_restructure() which uses self.direction_prob
        config = kwargs.get("config")
        self.direction_prob = getattr(config, "direction_prob", 0.0) if config else 0.0
        super().__init__(*args, **kwargs)

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for LVIS."""
        return tfds.builder("lvis:1.0.0", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "lvis"

    def get_num_transitions(self) -> int:
        """Return approximate number of LVIS samples."""
        return 1231766  # Approximate number of LVIS train images

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from LVIS image ID and annotation.

        Since the same image can have multiple object annotations (different categories
        or multiple instances of the same category), we need to include annotation-specific
        information to ensure uniqueness.
        """
        image_id = example["image_id"]
        category_name = example["annotations"]["category_name"][0]
        bbox = example["annotations"]["bbox"][0]

        # Create a hash from bbox coordinates to handle multiple instances of same category
        bbox_flat = tf.reshape(bbox, [-1])
        bbox_str = tf.strings.reduce_join(tf.strings.as_string(bbox_flat), separator="_")
        combined = tf.strings.join([image_id, "_", category_name, "_", bbox_str])
        combined_hash = tf.strings.to_hash_bucket_fast(combined, 2147483647)
        combined_hash_str = tf.strings.as_string(combined_hash)
        return tf.strings.join(["lvis_", image_id, "_", combined_hash_str])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from LVIS example.

        Returns:
            (prompt, caption) where prompt asks about the specific category
            and caption contains the formatted bbox or direction (50% each).
        """
        if self.directional:
            return self._extract_direction_prompt_and_caption(example)

        # Generate deterministic seed from image ID hash
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        # Extract the single category name (each example has one annotation)
        category_name = example["annotations"]["category_name"][0]

        # Extract the bbox (shape: [2, 2] with [[x_min, y_min], [x_max, y_max]])
        bbox = example["annotations"]["bbox"][0]

        # Sample a prompt using the shared helper
        prompt = sample_prompt_tf(ROBOT_BBOX_PROMPT_PARTS_OXE, category_name, (self.seed, image_id_hash))

        # With direction_prob probability, use direction caption instead of bbox
        dir_seed = (self.seed + 7919, image_id_hash)
        use_direction = tf.random.stateless_uniform([], seed=dir_seed, dtype=tf.float32) < self.direction_prob

        # Format bbox as caption using shared function
        bbox_caption = bbox_to_text_tf(bbox)

        # Get direction from bbox with "move " prefix
        direction_caption = direction_from_bbox_tf(bbox, slope=self.direction_slope, add_move_prefix=True)

        caption = tf.cond(use_direction, lambda: direction_caption, lambda: bbox_caption)

        return prompt, caption

    def _extract_direction_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt/caption asking for direction of the object relative to center."""
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        category_name = example["annotations"]["category_name"][0]
        bbox = example["annotations"]["bbox"][0]

        # Sample a direction prompt using the shared helper
        prompt = sample_prompt_tf(DIRECTION_PROMPT_PARTS, category_name, (self.seed, image_id_hash))

        # Get direction from bbox using shared function with "move " prefix
        direction_caption = direction_from_bbox_tf(bbox, slope=self.direction_slope, add_move_prefix=True)

        return prompt, direction_caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode LVIS image to JPEG bytes."""
        image = example["image"]
        # LVIS images are not pre-encoded, so encode them
        return tf.io.encode_jpeg(image, quality=95)
