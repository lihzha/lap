"""VQA dataset implementations.

Provides dataset classes for vision-language datasets like COCO Captions,
VQAv2, PixmoCap, PixmoPoint, LVIS, PACO, and bbox datasets.
"""

from lap.datasets.registry import VQA_DATASET_ID_MAP
from lap.datasets.registry import VQA_DATASET_ID_TO_NAME
from lap.datasets.registry import VQA_DATASET_NAMES
from lap.datasets.vqa.coco_caption_dataset import CocoCaption
from lap.datasets.vqa.lvis_dataset import Lvis
from lap.datasets.vqa.paco_dataset import PacoEgo4d
from lap.datasets.vqa.paco_dataset import PacoLvis
from lap.datasets.vqa.pixmo_cap_dataset import PixmoCap
from lap.datasets.vqa.pixmo_point_dataset import PixmoPoint
from lap.datasets.vqa.vqa_base import NUM_VQA_DATASETS
from lap.datasets.vqa.vqa_base import BaseVQADataset
from lap.datasets.vqa.vqav2_dataset import Vqav2

__all__ = [
    # Base
    "BaseVQADataset",
    "VQA_DATASET_ID_MAP",
    "VQA_DATASET_ID_TO_NAME",
    "VQA_DATASET_NAMES",
    "NUM_VQA_DATASETS",
    # VQA datasets
    "CocoCaption",
    "Vqav2",
    "PixmoCap",
    "PixmoPoint",
    "Lvis",
    "PacoLvis",
    "PacoEgo4d",
]
