"""DROID-specific mixins for shared lookup table functionality.

This module provides mixins for building and using DROID-specific
lookup tables (episode ID, filter, instruction tables) that are
shared between DroidDataset and DroidBoundingBoxDataset.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import tensorflow as tf

from lap.datasets.utils.helpers import extract_episode_path_from_file_path
from lap.datasets.utils.specs import RldsDatasetSpec
from lap.datasets.utils.tfdata_pipeline import print_memory_usage

if TYPE_CHECKING:
    pass


class DroidLookupTableMixin:
    """Mixin for DROID-specific lookup table building.

    This mixin provides methods for building the hash tables used
    by DROID datasets for episode identification, filtering, and
    instruction lookup.

    Usage:
        class MyDroidDataset(DroidLookupTableMixin, BaseRobotDataset):
            def __init__(self, ...):
                self._init_droid_tables(config, hash_tables)
                super().__init__(...)
    """

    # Class attribute to indicate this dataset uses hash tables
    REQUIRES_HASH_TABLES: bool = True

    spec: RldsDatasetSpec  # Will be provided by BaseRobotDataset

    def _init_droid_tables(
        self,
        config,
        hash_tables: dict | None,
        standalone: bool = False,
        build_filter_table: bool = True,
        build_instr_table: bool = True,
    ) -> None:
        """Initialize DROID lookup tables from config or provided hash_tables.

        Args:
            config: Data configuration with rlds_data_dir.
            hash_tables: Optional pre-built hash tables dict.
            standalone: If True and tables are built, store them in self.hash_tables.
            build_filter_table: Whether to build the filter table.
            build_instr_table: Whether to build the instruction table.
        """
        if hash_tables is not None:
            self.ep_table = hash_tables.get("ep_table")
            self.filter_table = hash_tables.get("filter_table")
            self.instr_table = hash_tables.get("instr_table")
        else:
            metadata_path = self._resolve_metadata_path(config.rlds_data_dir)

            self.ep_table = self._build_episode_table(metadata_path)
            self.filter_table = self._build_filter_table(metadata_path) if build_filter_table else None
            self.instr_table = self._build_instruction_table(metadata_path) if build_instr_table else None

            if standalone:
                self.hash_tables = {
                    "ep_table": self.ep_table,
                    "filter_table": self.filter_table,
                    "instr_table": self.instr_table,
                }

    def _resolve_metadata_path(self, rlds_data_dir: str) -> str:
        """Resolve metadata path from RLDS data directory.

        Args:
            rlds_data_dir: Path to RLDS data directory.
        Returns:
            Path to metadata directory.

        Raises:
            ValueError: If directory pattern not recognized.
        """
        return rlds_data_dir.replace("OXE", self.spec.metadata_path_name)

    def _build_episode_table(self, metadata_path: str) -> tf.lookup.StaticHashTable:
        """Build episode-path to episode-ID lookup table.

        Args:
            metadata_path: Path to metadata directory.

        Returns:
            StaticHashTable mapping episode paths to episode IDs.
        """
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.episode_id_to_path_file}", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.spec.default_ep_value,
        )
        print_memory_usage("After building ep_table")
        return ep_table

    def _build_filter_table(self, metadata_path: str) -> tf.lookup.StaticHashTable:
        """Build per-step filter table from keep_ranges file.

        Args:
            metadata_path: Path to metadata directory.

        Returns:
            StaticHashTable mapping frame keys to keep status.
        """
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.keep_ranges_file}", "r") as f:
            filter_dict = json.load(f)
        logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

        keys_tensor = []
        values_tensor = []

        for episode_key, ranges in filter_dict.items():
            for start, end in ranges:
                for t in range(start, end):
                    frame_key = f"{episode_key}--{t}"
                    keys_tensor.append(frame_key)
                    values_tensor.append(True)

        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
            default_value=False,
        )
        print_memory_usage("After building filter_table (per-step)")
        logging.info("Filter hash table initialized")

        return filter_table

    def _build_instruction_table(self, metadata_path: str) -> tf.lookup.StaticHashTable:
        """Build language instruction lookup table.

        Args:
            metadata_path: Path to metadata directory.

        Returns:
            StaticHashTable mapping episode IDs to serialized instructions.
        """
        _instr_keys_py = []
        _instr_vals_ser = []

        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.droid_language_annotations_file}", "r") as fp:
            language_annotations = json.load(fp)

        _instr_keys_py = list(language_annotations.keys())
        for _eid in _instr_keys_py:
            _v = language_annotations[_eid]
            _arr = [
                _v.get("language_instruction1", ""),
                _v.get("language_instruction2", ""),
                _v.get("language_instruction3", ""),
            ]
            _arr = [s for s in _arr if len(s) > 0]
            if len(_arr) == 0:
                _instr_vals_ser.append(b"")
            else:
                _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())

        _instr_keys = tf.constant(_instr_keys_py, dtype=tf.string)
        _instr_vals = tf.constant(_instr_vals_ser, dtype=tf.string)
        _instr_default = tf.constant(b"", dtype=tf.string)

        instr_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(_instr_keys, _instr_vals),
            default_value=_instr_default,
        )
        print_memory_usage("After building instr_table")
        return instr_table

    @staticmethod
    def episode_id_from_traj(traj: dict, ep_table: tf.lookup.StaticHashTable) -> tf.Tensor:
        """Extract episode ID from trajectory metadata.

        Args:
            traj: Trajectory dictionary with traj_metadata.
            ep_table: Episode lookup table.

        Returns:
            Episode ID tensor.
        """
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def get_step_filter_key(self, traj: dict, step_indices: tf.Tensor) -> tf.Tensor:
        """Build filter keys for step-level filtering.

        Args:
            traj: Trajectory dictionary with traj_metadata.
            step_indices: Tensor of step indices as strings.

        Returns:
            Tensor of filter keys.
        """
        return (
            traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
            + "--"
            + traj["traj_metadata"]["episode_metadata"]["file_path"]
            + "--"
            + step_indices
        )
