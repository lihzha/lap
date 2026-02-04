"""Gemma3 core modules (copied from official Gemma3 repo with minimal changes).

This file contains the essential Attention, FeedForward, and RMSNorm layers
from the official Gemma3 codebase, adapted for use in Pi0 MoE architecture.

Uses nn.scan for efficient layer creation (matching gemma.py structure).
Attention type (local/global) is passed as a scanned input instead of using layer_idx.

Original sources:
- _modules.py: Attention, FeedForward, Embedder, Block
- _layers.py: Einsum, RMSNorm
- _positional_embeddings.py: apply_rope

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# ============================================================================
# Constants
# ============================================================================
K_MASK = -2.3819763e38
GEMMA3_VOCAB_SIZE = 262_144

KVCache: TypeAlias = tuple[
    at.Int[at.Array, "l b"], at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


# ============================================================================
# Configuration & Factory
# ============================================================================

Variant = Literal[
    "gemma3_300m",
    "gemma3_1b",
    "gemma3_4b",
    "gemma3_12b",
    "gemma3_300m_26",
    "gemma3_300m_48",
    "gemma3_27b",
    "gemma3_300m_62",
]


@dataclasses.dataclass
class Config:
    width: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int

    # Gemma3 attention pattern
    sliding_window_size: int = 1024
    sliding_window_pattern: int = 6  # 5 local + 1 global = every 6th is global

    # Gemma3 rope
    rope_local_base_freq: float = 10000.0
    rope_global_base_freq: float = 1000000.0

    # Gemma3 normalization (qk-norm replaces softcapping)
    use_qk_norm: bool = True
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True

    transpose_gating_einsum: bool = True
    vocab_size: int = GEMMA3_VOCAB_SIZE
    lora_configs: dict = dataclasses.field(default_factory=dict)
    param_dtype: str = "bfloat16"

    def get_attention_type(self, layer_idx: int) -> Literal["global", "local"]:
        """Determine if layer uses global or local (sliding window) attention.

        Pattern: 5 local layers, then 1 global layer, repeating.
        Layer 0,1,2,3,4 = local; Layer 5 = global; Layer 6,7,8,9,10 = local; etc.
        """
        if (layer_idx + 1) % self.sliding_window_pattern == 0:
            return "global"
        return "local"

    def get_is_global_attn_array(self) -> jax.Array:
        """Pre-compute boolean array indicating global attention for each layer.

        Returns:
            Boolean array of shape [num_layers] where True = global attention.
        """
        return jnp.array([self.get_attention_type(i) == "global" for i in range(self.num_layers)])


def get_config(variant: Variant) -> Config:
    """Returns config dict for specified Gemma3 variant."""
    if variant == "gemma3_1b":
        return Config(
            width=1152,
            hidden_dim=6 * 1152,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=26,
            sliding_window_size=512,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_4b":
        return Config(
            width=2560,
            hidden_dim=2560 * 8 // 2,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=34,
            sliding_window_size=1024,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_12b":
        return Config(
            width=3840,
            hidden_dim=8 * 3840 // 2,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=48,
            sliding_window_size=1024,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_27b":
        return Config(
            width=5376,
            hidden_dim=5376 * 8 // 2,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=62,
            sliding_window_size=1024,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_300m":
        return Config(
            width=768,
            hidden_dim=768 * 4,
            num_heads=8,
            num_kv_heads=4,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=34,
            sliding_window_size=512,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_300m_26":
        return Config(
            width=768,
            hidden_dim=768 * 4,
            num_heads=4,
            num_kv_heads=1,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=26,
            sliding_window_size=512,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_300m_48":
        return Config(
            width=768,
            hidden_dim=768 * 4,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=48,
            sliding_window_size=512,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    if variant == "gemma3_300m_62":
        return Config(
            width=768,
            hidden_dim=768 * 4,
            num_heads=32,
            num_kv_heads=16,
            head_dim=128,
            vocab_size=GEMMA3_VOCAB_SIZE,
            num_layers=62,
            sliding_window_size=512,
            sliding_window_pattern=6,
            rope_local_base_freq=10000.0,
            rope_global_base_freq=1000000.0,
            use_qk_norm=True,
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            transpose_gating_einsum=True,
        )
    raise ValueError(f"Unknown variant: {variant}")


# ============================================================================
# Embedder
# ============================================================================


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


# ============================================================================
# RMSNorm
# ============================================================================


@at.typecheck
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Based on Gemma's implementation which uses (1 + weight) scaling.
    Supports adaptive RMSNorm with conditioning (adarms).
    """

    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))

        if cond is None:
            # regular RMSNorm with (1 + scale) like Gemma
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


class QKRMSNorm(nn.Module):
    """RMSNorm for Query/Key normalization in Gemma3 attention.

    Unlike the main RMSNorm, this uses direct scaling (not 1 + scale).
    Creates a nested 'scale' parameter to match checkpoint structure:
    - q_rmsnorm/scale
    - k_rmsnorm/scale
    """

    param_dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply RMSNorm to query or key tensors.

        Args:
            x: Input tensor of shape [..., head_dim]

        Returns:
            Normalized tensor of same shape
        """
        original_shape = x.shape
        head_dim = x.shape[-1]
        dtype = x.dtype

        # Flatten for normalization
        x_flat = x.reshape(-1, head_dim).astype(jnp.float32)

        # Scale parameter - initialized to ones for identity at start
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (head_dim,),
            jnp.dtype(self.param_dtype),
        )

        # RMSNorm computation
        variance = jnp.mean(x_flat**2, axis=-1, keepdims=True)
        x_normed = x_flat * jax.lax.rsqrt(variance + 1e-6)
        x_normed = x_normed * scale.astype(jnp.float32)

        return x_normed.reshape(original_shape).astype(dtype)


# ============================================================================
# Einsum Layer
# ============================================================================


class Einsum(nn.Module):
    """Einsum layer for parameterized tensor multiplication."""

    shape: tuple[int, ...]
    weight_name: str = "w"
    initializer: nn.initializers.Initializer = nn.initializers.normal()
    param_dtype: jnp.dtype | None = None
    lora_config: "lora.LoRAConfig" = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        if self.lora_config is not None:
            # Use LoRA if configured
            w = lora.Einsum(
                shape=self.shape,
                weight_name=self.weight_name,
                init_fn=self.initializer,
                lora_config=self.lora_config,
            )(eqn, x)
            return w
        w = self.param(
            self.weight_name,
            self.initializer,
            self.shape,
            self.param_dtype if self.param_dtype is not None else None,
        )
        return jnp.einsum(eqn, x, w)


# ============================================================================
# FeedForward
# ============================================================================


@at.typecheck
class FeedForward(nn.Module):
    """Gemma3MLP-compatible FeedForward using GeGLU activation."""

    features: int
    hidden_dim: int
    lora_config: "lora.LoRAConfig" = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype

        # Gating + Up projections fused into a single tensor
        gating = Einsum(
            shape=(2, self.hidden_dim, self.features),
            weight_name="gating_einsum",
            initializer=nn.initializers.lecun_normal(),
            param_dtype=dtype,
            lora_config=self.lora_config,
        )

        eq = "...F,GHF->...GH"
        gate_full = gating(eq, x)
        gate = gate_full[..., 0, :]
        up = gate_full[..., 1, :]

        # GeGLU activation
        hidden = jax.nn.gelu(gate) * up

        # Down projection
        linear = Einsum(
            shape=(self.hidden_dim, self.features),
            weight_name="linear",
            initializer=nn.initializers.lecun_normal(),
            param_dtype=dtype,
            lora_config=self.lora_config,
        )

        output = linear("...H,HF->...F", hidden)
        return output.astype(dtype)


# ============================================================================
# Positional Embeddings (RoPE)
# ============================================================================


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: float | jax.Array = 10000.0,
    scale_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE.

    Let B denote batch size, L denote sequence length, N denote number of heads,
    and H denote head dimension. Note that H must be divisible by 2.

    Args:
        inputs: Array of shape [B, L, N, H].
        positions: Array of shape [B, L].
        base_frequency: Base frequency used to compute rotations.
        scale_factor: The scale factor used for positional interpolation.

    Returns:
        Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]

    # Compute inverse frequencies
    dim_pairs = head_dim // 2
    freq_seq = jnp.arange(0, dim_pairs, dtype=jnp.float32)
    inv_freq = 1.0 / (base_frequency ** (freq_seq / dim_pairs))

    # Compute angles: [B, T, dim_pairs]
    positions = positions.astype(jnp.float32)
    angles = positions[:, :, None] * inv_freq[None, None, :]

    # Compute sin and cos
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)

    # Split x into even and odd components
    x1 = inputs[..., ::2]
    x2 = inputs[..., 1::2]

    # Expand sin/cos for broadcasting: [B, T, 1, dim_pairs]
    sin = sin[:, :, None, :]
    cos = cos[:, :, None, :]

    # Rotary embedding: [cos, -sin; sin, cos] @ [x1, x2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    out = jnp.stack([out1, out2], axis=-1).reshape(inputs.shape)
    return out.astype(inputs.dtype)


# ============================================================================
# Helper Functions
# ============================================================================


def _name(base_name: str, expert_idx: int) -> str:
    """Generate layer names for multi-expert setup.

    The first expert's weights have no suffix (for seamless loading from checkpoints).
    Subsequent experts get numeric suffixes and are initialized from scratch.

    Example: _name("attn", 0) -> "attn", _name("attn", 1) -> "attn_1"
    """
    if expert_idx == 0:
        return base_name
    return f"{base_name}_{expert_idx}"


def _gated_residual(x, y, gate):
    """Apply gated residual connection: x + y * gate (or x + y if no gate)."""
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate


# ============================================================================
# Attention
# ============================================================================


@at.typecheck
class Attention(nn.Module):
    """Gemma3 Attention with local/global sliding window support."""

    configs: Sequence[Config]
    is_global_attn: jax.Array | bool = False  # Whether this layer uses global attention (can be traced)
    stop_action_to_vlm_grad: bool = False
    cache_dtype: str | None = None

    @nn.compact
    def __call__(
        self,
        xs: Sequence[jnp.ndarray | None],
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        kv_cache: KVCache | None,
        image_mask: jnp.ndarray | None = None,
    ):
        # All experts must share the same head dim, num heads, and num kv heads
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        config = self.configs[0]
        dtype = next(x.dtype for x in xs if x is not None)

        # Determine RoPE base frequency based on attention type (use jnp.where for traced values)
        rope_base = jnp.where(self.is_global_attn, config.rope_global_base_freq, config.rope_local_base_freq)

        qkvs = []
        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if cfg.num_kv_heads == cfg.num_heads:
                qkv_einsum = Einsum(
                    shape=(3, cfg.num_heads, cfg.width, cfg.head_dim),
                    name=_name("qkv_einsum", i),
                    initializer=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=cfg.lora_configs.get("attn"),
                    param_dtype=cfg.param_dtype,
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:  # GQA
                q_einsum = Einsum(
                    shape=(cfg.num_heads, cfg.width, cfg.head_dim),
                    name=_name("q_einsum", i),
                    initializer=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=cfg.lora_configs.get("attn"),
                    param_dtype=cfg.param_dtype,
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = Einsum(
                    shape=(2, cfg.num_kv_heads, cfg.width, cfg.head_dim),
                    name=_name("kv_einsum", i),
                    initializer=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=cfg.lora_configs.get("attn"),
                    param_dtype=cfg.param_dtype,
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # Ensure q, k, v are in the expected dtype (einsum can promote to float32)
        q = q.astype(dtype)
        k = k.astype(dtype)
        v = v.astype(dtype)

        token_owner = None
        if self.stop_action_to_vlm_grad and xs[0] is not None:
            token_owner = []
            for i, x in enumerate(xs):
                if x is not None:
                    token_owner.append(jnp.full((x.shape[0], x.shape[1]), i, dtype=jnp.int32))
            token_owner = jnp.concatenate(token_owner, axis=1)

        # Apply QK-norm (Gemma3 replaces softcapping with this)
        if config.use_qk_norm:
            q = QKRMSNorm(param_dtype=config.param_dtype, name="q_rmsnorm")(q)
            k = QKRMSNorm(param_dtype=config.param_dtype, name="k_rmsnorm")(k)

        # Apply RoPE with layer-appropriate base frequency
        q = apply_rope(q, positions=positions, base_frequency=rope_base)
        k = apply_rope(k, positions=positions, base_frequency=rope_base)

        # Scale query (cast scalar to preserve dtype)
        q = q * jnp.asarray(config.head_dim**-0.5, dtype=dtype)

        # Should still be half-precision here
        assert q.dtype == k.dtype == v.dtype == dtype, (
            f"q.dtype: {q.dtype}, k.dtype: {k.dtype}, v.dtype: {v.dtype}, dtype: {dtype}, head_dim: {config.head_dim}"
        )

        # Determine cache dtype (use explicit cache_dtype if set, otherwise use compute dtype)
        effective_cache_dtype = jnp.dtype(self.cache_dtype) if self.cache_dtype else dtype

        if kv_cache is not None:  # inference time
            idx, cache_k, cache_v = kv_cache
            if xs[0] is not None:
                idx, k, v = _update_cache(k, v, idx, cache_k, cache_v, cache_dtype=effective_cache_dtype)
            else:
                idx += k.shape[1]
                # Ensure consistent dtype during concatenation
                k = jnp.concatenate([cache_k, k.astype(effective_cache_dtype)], axis=1)
                v = jnp.concatenate([cache_v, v.astype(effective_cache_dtype)], axis=1)
        else:
            idx, k, v = _init_cache(k, v, attn_mask.shape[-1], cache_dtype=effective_cache_dtype)

        # Compute attention logits
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=config.num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        if self.stop_action_to_vlm_grad and xs[0] is not None:
            q_owner = token_owner[:, None, None, :, None]
            k_owner = token_owner[:, None, None, None, :]
            cross_to_expert0 = (q_owner != 0) & (k_owner == 0)

            expert0_len = xs[0].shape[1]
            k0 = k[:, :expert0_len, ...]
            q_i = q[:, expert0_len:, ...]
            logits0_i = jnp.einsum(
                "BTKGH,BSKH->BKGTS", q_i, jax.lax.stop_gradient(k0), preferred_element_type=jnp.float32
            )
            logits = logits.at[:, :, :, expert0_len:, :expert0_len].set(logits0_i)

        effective_mask = attn_mask

        # Apply sliding window mask for local attention (always compute, conditionally apply)
        sliding_mask = self._compute_sliding_window_mask(positions, k.shape[1], config.sliding_window_size)
        # For global attention, use original mask; for local, apply sliding window
        effective_mask = jnp.where(
            self.is_global_attn,
            effective_mask,  # global: keep original mask
            effective_mask & sliding_mask,  # local: apply sliding window
        )

        # Apply bidirectional attention for image tokens
        if image_mask is not None:
            effective_mask = self._apply_bidirectional_image_attention(effective_mask, image_mask)

        # Validate mask shape
        if effective_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {effective_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # Apply mask
        masked_logits = jnp.where(effective_mask[:, :, None, :, :], logits, K_MASK)

        # Softmax and weighted sum
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        if self.stop_action_to_vlm_grad and xs[0] is not None:
            probs_cross = probs * cross_to_expert0.astype(probs.dtype)
            probs_self = probs - probs_cross
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_self, v) + jnp.einsum(
                "BKGTS,BSKH->BTKGH", probs_cross, jax.lax.stop_gradient(v)
            )
        else:
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # Output projection for each expert
        out = []
        start = 0
        for i, (x, cfg) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = Einsum(
                    shape=(cfg.num_heads, cfg.head_dim, cfg.width),
                    name=_name("attn_vec_einsum", i),
                    initializer=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=cfg.lora_configs.get("attn"),
                    param_dtype=cfg.param_dtype,
                )
                # Cast output back to input dtype to ensure consistent dtypes through scan
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]).astype(dtype))
                start = end
            else:
                out.append(None)

        return out, (idx, k, v)

    def _compute_sliding_window_mask(
        self,
        positions: jnp.ndarray,
        kv_len: int,
        window_size: int,
    ) -> jnp.ndarray:
        """Compute sliding window attention mask."""
        B, T = positions.shape
        q_pos = positions[:, :, None]
        k_pos = jnp.arange(kv_len)[None, None, :]

        causal_mask = k_pos <= q_pos
        window_mask = (q_pos - k_pos) < window_size
        combined = causal_mask & window_mask

        return combined[:, None, :, :]

    def _apply_bidirectional_image_attention(
        self,
        attn_mask: jnp.ndarray,
        image_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Allow image tokens to attend to each other bidirectionally."""
        B, _, T, S = attn_mask.shape

        is_image_query = image_mask[:, None, :, None]
        is_image_key = image_mask[:, None, None, :S]

        bidirectional_region = is_image_query & is_image_key
        return attn_mask | bidirectional_region


# ============================================================================
# Block
# ============================================================================


@at.typecheck
class Block(nn.Module):
    """Transformer block with Gemma3 features."""

    configs: tuple[Config, ...]
    stop_action_to_vlm_grad: bool = False
    cache_dtype: str | None = None

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs: Sequence[jnp.ndarray | None],
        kv_cache: KVCache | None,
        positions: jnp.ndarray,
        attn_mask: jnp.ndarray,
        adarms_cond: Sequence[jnp.ndarray | None],
        is_global_attn: jax.Array | bool,  # Passed from scan (can be traced)
        image_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # Pre-attention normalization
        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
            else:
                gate = None
            pre_attn.append(x)
            gates.append(gate)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)

        # Attention with is_global_attn for local/global determination
        attn = Attention(
            configs=self.configs,
            is_global_attn=is_global_attn,
            stop_action_to_vlm_grad=self.stop_action_to_vlm_grad,
            cache_dtype=self.cache_dtype,
            name="attn",
        )
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, image_mask)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)

        # Post-attention norm (Gemma3 feature - only for first expert)
        if self.configs[0].use_post_attn_norm:
            post_attn_normed = []
            for i, x in enumerate(post_attn):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_attention_norm")(x, None)
                post_attn_normed.append(x)
            post_attn = post_attn_normed

        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # Feed-forward
        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
                x = FeedForward(
                    features=config.width,
                    hidden_dim=config.hidden_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            else:
                gate = None
            out.append(x)
            gates.append(gate)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)

        # Post-FFW norm (Gemma3 feature - only for first expert)
        if self.configs[0].use_post_ffw_norm:
            out_normed = []
            for i, x in enumerate(out):
                if x is not None and i == 0:
                    x, _ = RMSNorm(name="post_ffw_norm")(x, None)
                out_normed.append(x)
            out = out_normed

        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


# ============================================================================
# Module (Main Transformer)
# ============================================================================


@at.typecheck
class Module(nn.Module):
    """Transformer model supporting mixture of experts for Pi0.

    Uses nn.scan for efficient layer creation. Attention type (local/global)
    is pre-computed and passed as a scanned input to each layer.
    """

    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    stop_action_to_vlm_grad: bool = False

    cache_dtype: str | None = None

    def setup(self):
        assert all(config.num_layers == self.configs[0].num_layers for config in self.configs)

        self.embedder = Embedder(
            vocab_size=self.configs[0].vocab_size,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        # Pre-compute attention types for all layers
        self._is_global_attn = self.configs[0].get_is_global_attn_array()

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(6, 7),  # 0=self, 6=image_mask (can be None), 7=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        # Use nn.scan for efficient layer creation (like gemma.py)
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,  # kv_cache - stacked per layer
                nn.broadcast,  # positions
                nn.broadcast,  # attn_mask
                nn.broadcast,  # adarms_cond
                0,  # is_global_attn - scanned per layer
                nn.broadcast,  # image_mask
                nn.broadcast,  # deterministic
            ),
            length=self.configs[0].num_layers,
        )(
            configs=tuple(self.configs),
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            stop_action_to_vlm_grad=self.stop_action_to_vlm_grad,
            cache_dtype=self.cache_dtype,
        )

        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # @at.typecheck
    def decode(self, prelogits: at.Int[at.Array, "b t d"]) -> at.Float[at.Array, "b t"]:
        return self.embedder.decode(prelogits)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        image_mask: at.Bool[at.Array, "b t"] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # Pass is_global_attn array as scanned input
        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, self._is_global_attn, image_mask, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        out = [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ]
        return out, kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# ============================================================================
# KV Cache Utilities
# ============================================================================


def _init_cache(k, v, cache_size, cache_dtype=None):
    """Initialize KV cache."""
    prefill_len = k.shape[1]
    pad_width = ((0, 0), (0, cache_size - prefill_len), (0, 0), (0, 0))
    cache_dtype = cache_dtype or k.dtype
    k_cache = jnp.pad(k.astype(cache_dtype), pad_width)
    v_cache = jnp.pad(v.astype(cache_dtype), pad_width)
    idx = jnp.zeros((k.shape[0],), dtype=jnp.int32) + prefill_len
    return idx, k_cache, v_cache


def _update_cache(k, v, idx, k_cache, v_cache, cache_dtype=None):
    """Update KV cache with new values."""
    assert k.shape[1] == 1, "Only support kv-cache updates of length 1"
    indices = (0, idx[0], 0, 0)
    cache_dtype = cache_dtype or k.dtype
    k_new = jax.lax.dynamic_update_slice(k_cache, k.astype(cache_dtype), indices)
    v_new = jax.lax.dynamic_update_slice(v_cache, v.astype(cache_dtype), indices)
    idx_new = idx + 1
    return idx_new, k_new, v_new
