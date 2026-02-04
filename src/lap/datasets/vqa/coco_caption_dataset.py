"""COCO Captions dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.registry import register_dataset
from lap.datasets.vqa.vqa_base import BaseVQADataset

COCO_CAPTION_PROMPTS = tf.constant(
    [
        "Caption the image.",
        "Give a short caption.",
        "Provide a brief description.",
        "What is shown?",
        "Summarize the image in a few words.",
        "Describe it concisely.",
        "One-sentence caption, please.",
        "Give a minimal caption.",
        "What's happening?",
        "A short description.",
        "Describe this briefly.",
        "Caption in one phrase.",
        "What is depicted?",
        "Label the image content.",
        "Provide a simple caption.",
        "In a few words, what is this?",
        "Write a concise caption.",
        "What does the picture show?",
        "Give a very short image description.",
        "Provide a compact caption.",
    ],
    dtype=tf.string,
)


@register_dataset(name="coco_captions", is_vqa=True)
class CocoCaption(BaseVQADataset):
    """COCO Captions dataset for vision-language training.

    This dataset loads COCO images with captions and formats them to be
    compatible with the robot dataset structure.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for COCO captions."""
        return tfds.builder("coco_captions", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "coco_captions"

    def get_num_transitions(self) -> int:
        """Return approximate number of COCO caption samples."""
        return 80340  # Approximate number of COCO train images

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from COCO image filename and ID."""
        image_filename = example["image/filename"]
        image_id = tf.strings.as_string(example["image/id"])
        return tf.strings.join(["coco_", image_filename, "_", image_id])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from COCO example.

        Returns:
            (prompt, caption) where prompt is randomly sampled from COCO_PROMPTS
            and caption is randomly sampled from the available captions.
        """
        # Generate deterministic seed from image ID
        image_id = example["image/id"]
        image_id_hash = tf.cast(image_id % 2147483647, tf.int32)

        # Randomly select one caption from the list
        captions = example["captions"]["text"]
        num_captions = tf.shape(captions)[0]
        caption_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, image_id_hash], minval=0, maxval=num_captions, dtype=tf.int32
        )
        selected_caption = captions[caption_idx]

        # Randomly select a prompt
        num_prompts = tf.shape(COCO_CAPTION_PROMPTS)[0]
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed + 1, image_id_hash], minval=0, maxval=num_prompts, dtype=tf.int32
        )
        prompt = COCO_CAPTION_PROMPTS[prompt_idx]

        return prompt, selected_caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode COCO image to JPEG bytes."""
        image = example["image"]
        return tf.io.encode_jpeg(image, quality=95)
