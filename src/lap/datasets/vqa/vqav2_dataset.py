"""VQAv2 dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from lap.datasets.registry import register_dataset
from lap.datasets.vqa.vqa_base import BaseVQADataset


@register_dataset(name="vqa", is_vqa=True)
class Vqav2(BaseVQADataset):
    """VQAv2 dataset for vision-language training.

    This dataset loads VQAv2 images with questions and answers and formats them
    to be compatible with the robot dataset structure.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for VQAv2.

        Note: This assumes the vqav2.py file has been registered with TFDS
        and the dataset builder is named 'vqa'.
        """

        return tfds.builder("vqa", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "vqa"

    def get_num_transitions(self) -> int:
        """Return approximate number of VQAv2 samples."""
        return 443800

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from VQAv2 question_id and image_id."""
        question_id = tf.strings.as_string(example["question_id"])
        image_id = tf.strings.as_string(example["image/id"])
        return tf.strings.join(["vqa_", question_id, "_", image_id])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from VQAv2 example.

        Returns:
            (prompt, caption) where prompt is the question_text
            and caption is the top_answer.
        """
        prompt = example["question_text"]
        caption = example["top_answer"]
        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode VQAv2 image to JPEG bytes."""
        image = example["image"]

        # Check if image is already encoded (bytes) or needs encoding
        if image.dtype == tf.string:
            # Already encoded
            return image
        # Encode image to bytes
        return tf.io.encode_jpeg(image, quality=95)
