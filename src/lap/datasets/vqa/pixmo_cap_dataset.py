"""PixmoCap dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.registry import register_dataset
from lap.datasets.vqa.vqa_base import BaseVQADataset

# PixmoCap prompts to randomly sample from
PIXMO_CAP_PROMPTS = tf.constant(
    [
        "Describe this image.",
        "Describe this image",
        "describe the image",
        "Write a long description of this image.",
        "caption the picture",
        "Caption",
        "caption",
        "Construct a long caption for this image",
        "Generate a caption",
        "Create a detailed caption",
        "Write a long caption",
        "Describe this image in detail",
        "Describe this",
        "describe this",
        "Caption this",
        "What can be seen in this image?",
        "What do you see in the image?",
        "Look at this photo carefully and then tell me about it in detail",
        "Write a long description of this image",
        "Tell me about this picture.",
        "Write a paragraph about this image.",
        "Look at this image carefully and then describe it in detail",
        "Generate a long caption about this image.",
        "Describe this image in detail, but without any pointing.",
        "Write a long description of this image, do not produce any points.",
        "Tell me about this picture, use plain text only.",
        "Generate a plain text description of this caption",
        "What is in this image?\nNo pointing\nGive lots of detail"
        "Write a long caption.\nDo not use image coordinates\nOutput a full paragraph",
    ],
    dtype=tf.string,
)


@register_dataset(name="pixmo_cap", is_vqa=True)
class PixmoCap(BaseVQADataset):
    """PixmoCap dataset for vision-language training.

    This dataset loads PixmoCap images with captions and formats them to be
    compatible with the robot dataset structure.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for PixmoCap."""

        return tfds.builder("pixmo_cap", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "pixmo_cap"

    def get_num_transitions(self) -> int:
        """Return approximate number of PixmoCap samples."""
        return 620032

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from PixmoCap image filename and caption.

        Since the same image can have multiple captions, we need to include
        both the image filename and caption to ensure uniqueness.
        """
        image_filename = example["image_filename"]
        caption = example["caption"]
        # Create a hash of the combination to ensure uniqueness
        combined = tf.strings.join([image_filename, "_", caption])
        combined_hash = tf.strings.to_hash_bucket_fast(combined, 2147483647)
        combined_hash_str = tf.strings.as_string(combined_hash)
        return tf.strings.join(["pixmo_cap_", image_filename, "_", combined_hash_str])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from PixmoCap example.

        Returns:
            (prompt, caption) where prompt is randomly sampled from PIXMO_CAP_PROMPTS
            and caption is the single caption from the dataset.
        """
        # Generate deterministic seed from image filename hash
        image_filename = example["image_filename"]
        filename_hash = tf.strings.to_hash_bucket_fast(image_filename, 2147483647)
        filename_hash = tf.cast(filename_hash, tf.int32)

        # Get the caption (single string, not a list)
        caption = example["caption"]

        # Randomly select a prompt
        num_prompts = tf.shape(PIXMO_CAP_PROMPTS)[0]
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, filename_hash], minval=0, maxval=num_prompts, dtype=tf.int32
        )
        prompt = PIXMO_CAP_PROMPTS[prompt_idx]

        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode PixmoCap image to JPEG bytes."""
        image = example["image"]
        # Check if image is already encoded (has dtype string)
        if image.dtype == tf.string:
            # Already encoded, return as-is
            return image
        # Not encoded, encode to JPEG
        return tf.io.encode_jpeg(image, quality=95)
