"""Image collection and masking utilities for CoT policy."""

import dataclasses

import numpy as np

from lap.models.model_adapter import IMAGE_KEYS
from lap.policies.transforms.image_utils import parse_image


@dataclasses.dataclass
class ImageHandler:
    """Handles image collection, masking, and preprocessing for policy inputs.

    Attributes:
        wrist_image_dropout_prob: Probability of dropping wrist images during training.
        random_mask_prob: Probability of masking zero images (for data augmentation).
    """

    wrist_image_dropout_prob: float = 0.0
    random_mask_prob: float = 0.0

    @staticmethod
    def create_image_mask(image: np.ndarray, random_mask_prob: float = 0.0) -> np.bool_:
        """Create a mask indicating whether an image should be used.

        Args:
            image: Input image array.
            random_mask_prob: Probability of randomly masking zero images.

        Returns:
            Boolean mask (True = use image, False = mask out).
        """
        if np.all(image == 0.0):
            if random_mask_prob > 0.0 and np.random.rand() < random_mask_prob:
                return np.True_
            return np.False_
        return np.True_

    def collect_images(
        self,
        data: dict,
        base_image: np.ndarray,
        *,
        is_prediction_sample: bool = False,
        pred_use_primary: bool = False,
        is_vqa_sample: bool = False,
    ) -> tuple[list[np.ndarray], list[np.bool_]]:
        """Collect and process images for model input.

        Handles three cases:
        1. Regular samples: Use base image + wrist images with dropout
        2. Prediction samples without primary: All images without modification
        3. Prediction samples with primary: Similar to regular samples

        Note: Wrist image rotation is handled at the dataset level.

        Args:
            data: Data dictionary containing observation images.
            base_image: The primary/base camera image.
            is_prediction_sample: Whether this is a prediction training sample.
            pred_use_primary: Whether prediction sample should use primary image processing.
            is_vqa_sample: Whether this is a VQA sample (disables random augmentation).

        Returns:
            Tuple of (images_list, image_masks_list).
        """
        images: list[np.ndarray] = []
        image_masks: list[np.bool_] = []

        observation = data.get("observation", {})

        def add_image(image: np.ndarray, random_mask_prob: float = 0.0) -> None:
            mask = self.create_image_mask(image, random_mask_prob=random_mask_prob)
            images.append(image)
            image_masks.append(mask)

        def get_effective_mask_prob() -> float:
            """Get mask probability, disabled for VQA samples."""
            return 0.0 if is_vqa_sample else self.random_mask_prob

        if not is_prediction_sample:
            # Regular sample processing
            add_image(base_image)
            for key in IMAGE_KEYS[1:]:
                wrist_image = self._get_wrist_image(observation, key, base_image, is_vqa_sample)
                add_image(wrist_image, random_mask_prob=get_effective_mask_prob())

        elif not pred_use_primary:
            # Prediction sample without primary - use all images as-is
            for key in IMAGE_KEYS:
                if key in observation:
                    image = parse_image(observation[key])
                    add_image(image)
                else:
                    add_image(np.zeros_like(base_image))

        else:
            # Prediction sample with primary
            add_image(base_image)
            for key in IMAGE_KEYS[1:]:
                if key in observation:
                    wrist_image = parse_image(observation[key])
                    add_image(wrist_image)
                else:
                    add_image(np.zeros_like(base_image))

        return images, image_masks

    def _get_wrist_image(
        self,
        observation: dict,
        key: str,
        base_image: np.ndarray,
        is_vqa_sample: bool,
    ) -> np.ndarray:
        """Get wrist image with optional dropout.

        Args:
            observation: Observation dictionary.
            key: Image key to retrieve.
            base_image: Base image for shape reference.
            is_vqa_sample: Whether this is a VQA sample (disables dropout).

        Returns:
            Wrist image array.
        """
        if key not in observation:
            return np.zeros_like(base_image)

        wrist_image = parse_image(observation[key])

        # Apply dropout (disabled for VQA samples)
        should_dropout = (
            not is_vqa_sample
            and self.wrist_image_dropout_prob > 0.0
            and np.random.rand() < float(self.wrist_image_dropout_prob)
        )

        if should_dropout:
            return np.zeros_like(base_image)

        return wrist_image

    @staticmethod
    def parse_base_image(data: dict) -> np.ndarray:
        """Parse the base image from data, handling edge cases.

        Args:
            data: Data dictionary containing observation.

        Returns:
            Base image as numpy array.
        """
        base_image_raw = data["observation"].get(IMAGE_KEYS[0])

        # Handle empty string (for bbox samples using only wrist image)
        if isinstance(base_image_raw, (str, bytes)) and len(base_image_raw) == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        base_image = parse_image(base_image_raw)
        if base_image is None:
            # Create a zero image as fallback (will be masked out)
            return np.zeros((224, 224, 3), dtype=np.uint8)

        return base_image
