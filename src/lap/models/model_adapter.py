from collections.abc import Sequence
import dataclasses
from enum import Enum
import logging
from typing import Generic, TypeVar

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.shared import image_tools
import openpi.shared.array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray)

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)

logger = logging.getLogger("openpi")


class ExtendedModelType(str, Enum):
    """All model types, including CoT extensions."""

    PI0 = _model.ModelType.PI0.value
    PI0_FAST = _model.ModelType.PI0_FAST.value
    PI05 = _model.ModelType.PI05.value
    LAP = "lap"
    LAP_FAST = "lap_fast"


@at.typecheck
@struct.dataclass
class CoTObservation(_model.Observation[ArrayT], Generic[ArrayT]):
    # --- CoT / vis extras (all optional) ---
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    tokenized_langact_mask: at.Bool[ArrayT, "*b l"] | None = None
    critical_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    number_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    direction_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    sample_mask: at.Bool[ArrayT, "*b"] | None = None
    tokenized_dataset_name: at.Int[ArrayT, "*b d"] | None = None
    is_vqa_sample: at.Bool[ArrayT, "*b"] | None = None
    is_prediction_sample: at.Bool[ArrayT, "*b"] | None = None
    vqa_dataset_id: at.Int[ArrayT, "*b"] | None = None  # ID for per-VQA-dataset metrics (0=non-VQA)

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "CoTObservation[ArrayT]":
        # Build the base Observation first (handles images, masks, dtype fixes, etc.)
        data_dict = dict(data)

        base: _model.Observation[ArrayT] = _model.Observation.from_dict(data_dict)

        # Pull CoT extras from either flat keys or a namespaced location.
        cot_src = data.get("extras", {}).get("cot", {})

        # Allow flat keys as well:
        def getk(k):
            return data.get(k, cot_src.get(k, None))

        # Construct subclass using base fields
        base_dict = dataclasses.asdict(base)

        return cls(
            **base_dict,
            tokenized_langact_mask=getk("tokenized_langact_mask"),
            critical_token_mask=getk("critical_token_mask"),
            number_token_mask=getk("number_token_mask"),
            direction_token_mask=getk("direction_token_mask"),
            sample_mask=getk("sample_mask"),
            tokenized_dataset_name=getk("tokenized_dataset_name"),
            is_vqa_sample=getk("is_vqa_sample"),
            is_prediction_sample=getk("is_prediction_sample"),
            vqa_dataset_id=getk("vqa_dataset_id"),
        )


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: CoTObservation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
    aug_wrist_image: bool = True,
    enable_image_augmentation: bool = True,
    vqa_mask: at.Bool[jax.Array, "*b"] | None = None,
) -> CoTObservation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).

    Args:
        enable_image_augmentation: When False, disables all image augmentation even during training.
    """

    # if not set(image_keys).issubset(observation.images):
    #     raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        b, h, w, c = image.shape

        # Resize if needed (before augmentation)
        if (h, w) != image_resolution:
            logger.info(f"Resizing image {key} from {(h, w)} to {image_resolution}")
            # Process each frame
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train and enable_image_augmentation:
            # Augmentation: apply to each frame independently
            # Flatten: [b*t, h, w, c]
            h_new, w_new = image_resolution

            # Convert to [0, 1]
            image = image / 2.0 + 0.5

            # Build transforms
            transforms = []
            if "wrist" in key and aug_wrist_image:
                transforms += [
                    augmax.RandomCrop(int(w_new * 0.95), int(h_new * 0.95)),
                    augmax.Resize(w_new, h_new),
                    augmax.Rotate((-5, 5)),
                ]
            else:
                transforms += [
                    augmax.RandomCrop(int(w_new * 0.95), int(h_new * 0.95)),
                    augmax.Resize(w_new, h_new),
                    augmax.Rotate((-5, 5)),
                ]
            # transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            transforms += [augmax.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]

            # Apply augmentation
            sub_rngs = jax.random.split(rng, b)
            image_aug = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Skip augmentation for VQA samples if vqa_mask is provided
            image = jnp.where(vqa_mask[:, None, None, None], image, image_aug) if vqa_mask is not None else image_aug

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return CoTObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        tokenized_langact_mask=getattr(observation, "tokenized_langact_mask", None),
        critical_token_mask=getattr(observation, "critical_token_mask", None),
        number_token_mask=getattr(observation, "number_token_mask", None),
        direction_token_mask=getattr(observation, "direction_token_mask", None),
        sample_mask=getattr(observation, "sample_mask", None),
        tokenized_dataset_name=getattr(observation, "tokenized_dataset_name", None),
        is_vqa_sample=getattr(observation, "is_vqa_sample", None),
        is_prediction_sample=getattr(observation, "is_prediction_sample", None),
        vqa_dataset_id=getattr(observation, "vqa_dataset_id", None),
    )
