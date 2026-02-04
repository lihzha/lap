import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import openpi.policies.policy as _policy
from openpi.shared import nnx_utils

from lap.models.model_adapter import CoTObservation


class ARPolicy:
    """A policy that supports autoregressive sampling."""

    def __init__(self, base: _policy.Policy, *, sample_kwargs: dict[str, Any] | None = None):
        self._base = base
        assert hasattr(base._model, "sample_tokens"), "Model must have a sample_tokens method"  # noqa: SLF001
        self._sample_tokens = nnx_utils.module_jit(base._model.sample_tokens)
        # self._sample_tokens = base._model.sample_tokens

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def infer_reasoning(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        raw_state = inputs["observation"]["state"].copy()
        inputs = self._base._input_transform(inputs)  # noqa: SLF001
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng_or_pytorch_device = jax.random.split(self._base._rng)

        start_time = time.monotonic()
        tokens = self._sample_tokens(sample_rng_or_pytorch_device, CoTObservation.from_dict(inputs))
        outputs = {
            "state": inputs["state"],
            "tokens": tokens,
            "raw_state": raw_state,
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        # outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._base._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        return self.infer_reasoning(obs)

    def vqa_infer(self, obs: dict) -> dict:
        """Run VQA inference using the VLM backbone.

        Expects `obs` to contain images and a tokenized prompt compatible with the
        PaliGemma tokenizer. Returns generated text.
        """

        return self.infer_reasoning(obs)
