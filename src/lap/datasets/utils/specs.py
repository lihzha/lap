"""Data specifications and constants for RLDS datasets."""

from dataclasses import dataclass
from dataclasses import field

import tensorflow as tf


@dataclass(frozen=True)
class RldsDatasetSpec:
    metadata_path_name: str = "metadata"
    episode_id_to_path_file: str = "episode_id_to_path.json"
    cam2base_extrinsics_file: str = "cam2base_extrinsics.json"
    camera_serials_file: str = "camera_serials.json"
    intrinsics_file: str = "intrinsics.json"
    droid_language_annotations_file: str = "droid_language_annotations.json"
    keep_ranges_file: str = "keep_ranges_1_0_1.json"
    images_list: tuple[str, str] = ("exterior_image_1_left", "exterior_image_2_left")
    primary_image_key: str = "base_0_rgb"
    wrist_image_key: str = "left_wrist_0_rgb"
    wrist_image_right_key: str = "right_wrist_0_rgb"
    default_lang_value: bytes = field(
        default_factory=lambda: tf.io.serialize_tensor(tf.constant([], dtype=tf.string)).numpy()
    )
    default_ep_value: tf.Tensor = field(default_factory=lambda: tf.constant("", dtype=tf.string))
