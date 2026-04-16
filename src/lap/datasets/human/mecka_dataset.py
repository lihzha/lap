"""Mecka bimanual human hand dataset implementation.

Same egocentric bimanual structure as the Aria dataset, with an additional
per-step ``subtask`` field (dense segment-level annotation).  When ``subtask``
is non-empty it overrides the global ``language_instruction`` so the model
receives the finest-grained instruction available.

The dataset is split into 7 parts: mecka_dataset_part1 … mecka_dataset_part7.
"""

from __future__ import annotations

from lap.datasets.human.aria_dataset import AriaDataset
from lap.datasets.registry import register_dataset


@register_dataset(matcher=lambda n: "mecka" in n, priority=10)
class MeckaDataset(AriaDataset):
    """Dataset for Mecka bimanual human egocentric manipulation data.

    Identical processing to AriaDataset.  The subtask-based instruction
    override is handled in the standardization transform
    (mecka_dataset_transform in transforms.py) before the pipeline runs.
    """
