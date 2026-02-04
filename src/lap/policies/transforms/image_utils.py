"""Image parsing helpers used by policy transforms."""

import einops
import numpy as np


def parse_image(image) -> np.ndarray:
    if image is None:
        return None
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[1] == 3 and len(image.shape) == 4:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image

