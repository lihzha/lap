from collections.abc import Callable
from typing import Any

import jax
from jax.experimental import multihost_utils as mh
import numpy as np
from openpi.shared import array_typing as at


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging."""
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def to_local_array(x):
    """Return a NumPy view/copy of the process-local portion of a jax.Array."""
    if x is None:
        return None

    if not isinstance(x, jax.Array):
        try:
            return np.asarray(x)
        except Exception:
            return x

    try:
        if getattr(x, "is_fully_addressable", False):
            return np.asarray(x.block_until_ready())
    except Exception:
        pass

    shards = getattr(x, "addressable_shards", None)
    if shards:
        a0 = np.asarray(shards[0].data.block_until_ready())
        if a0.ndim == 0:
            return a0
        parts = [np.asarray(s.data.block_until_ready()) for s in shards]
        return np.concatenate(parts, axis=0)

    return x


def to_local_scalar(x) -> int:
    """Extract a Python scalar from a possibly-global jax.Array, process-local only."""
    if x is None:
        return 0
    try:
        shards = getattr(x, "addressable_shards", None)
        if shards is not None and len(shards) > 0:
            return int(np.asarray(shards[0].data).item())
    except Exception:
        pass
    return int(np.asarray(x).item())


def global_concat(local_values: np.ndarray) -> np.ndarray:
    """Gather and concatenate numpy arrays across all hosts, returning global view."""
    if local_values is None:
        return np.array([], dtype=np.float32)
    local_values = np.asarray(local_values)
    process_count = getattr(jax, "process_count", lambda: 1)()
    if process_count == 1:
        return local_values
    gathered = mh.process_allgather(local_values, tiled=False)
    return np.concatenate([np.asarray(x) for x in gathered], axis=0)
