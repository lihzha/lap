"""Dataset discovery and registration helpers."""

import importlib


def ensure_datasets_registered() -> None:
    """Import dataset modules so registration decorators run."""
    importlib.import_module("lap.datasets.robot.droid_dataset")
    importlib.import_module("lap.datasets.robot.oxe_datasets")

    importlib.import_module("lap.datasets.vqa.coco_caption_dataset")
    importlib.import_module("lap.datasets.vqa.lvis_dataset")
    importlib.import_module("lap.datasets.vqa.paco_dataset")
    importlib.import_module("lap.datasets.vqa.pixmo_cap_dataset")
    importlib.import_module("lap.datasets.vqa.pixmo_point_dataset")
    importlib.import_module("lap.datasets.vqa.vqav2_dataset")
