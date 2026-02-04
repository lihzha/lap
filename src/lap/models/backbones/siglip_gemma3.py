# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

"""ADDING RMSNORM FOR GEMMA3"""

from collections.abc import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import openpi.training.sharding as sharding

from lap.models.backbones.gemma3 import RMSNorm as CommonRMSNorm

# from lap.models.gemma_common import RMSNorm as CommonRMSNorm


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


def patchify_images(
    images: jnp.ndarray,
    *,
    patch_size: tuple[int, int],
    padding: str = "VALID",
) -> jnp.ndarray:
    """Extract patches from images.

    This function is a wrapper for jax.lax.conv_general_dilated_patches
    to conform to the same interface as tf.image.extract_patches.
    The function extracts patches of shape sizes from the input images in the same
    manner as a convolution with kernel of shape sizes, stride equal to strides,
    and the given padding scheme.
    The patches are stacked in the channel dimension.

    Args:
        images: input batch of images of shape [B, H, W, C].
        patch_size: size of extracted patches.
        padding: padding algorithm to use.

    Returns:
        Tensor of shape [batch, num patches, patch_size * patch_size * C]
    """
    channels = images.shape[-1]
    patches = jax.lax.conv_general_dilated_patches(
        lhs=images,
        filter_shape=patch_size,
        window_strides=patch_size,
        padding=padding,
        rhs_dilation=[1, 1],
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
        precision=jax.lax.Precision.HIGH,
    )
    patches = einops.rearrange(patches, "b ph pw (c p) -> b (ph pw) (p c)", c=channels)
    return patches


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "none"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    posemb_shape: tuple[int, int] | None = None
    gemma3_pooling: bool = True

    """ @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # ADD: Handle Pan & Scan multi-crop input
        original_shape = image.shape
        if len(image.shape) == 5:  # [B, N_crops, H, W, C]
            B, N, H, W, C = image.shape
            image = image.reshape(B * N, H, W, C)  # Flatten crops
            is_multicrop = True
        else:
            is_multicrop = False
            B = image.shape[0]

        *batch_dims, _, _, _ = image.shape
        image = einops.rearrange(image, "... h w c -> (...) h w c")

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction via Conv - automatically handles any image size (224×224 or 896×896)
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        seqshape = self.posemb_shape or (h, w)
        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, seqshape, c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # ====================================================================
        # CORRECT LOCATION FOR THE SOFTNORM LAYER
        # ====================================================================
        # The layer name "mm_soft_embedding_norm" will cause Flax to create
        # a parameter path that matches your checkpoint.
        x, _ = CommonRMSNorm(name="mm_soft_embedding_norm")(x, None)
        # ====================================================================

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        # Gemma3 Pooling
        if self.gemma3_pooling:
            n, seq_len, c = x.shape
            grid_size = int(seq_len ** 0.5)  # sqrt(h*w)
            assert grid_size * grid_size == seq_len, f"Sequence length {seq_len} must be perfect square"

            x_2d = x.reshape(n, grid_size, grid_size, c)  # [n, h, w, c]
            # 4×4 average pooling if grid is 64×64
            if grid_size == 64:
                # Pool 64×64 → 16×16
                x_2d_pooled = nn.avg_pool(x_2d, window_shape=(4, 4), strides=(4, 4))
                x = x_2d_pooled.reshape(n, 16*16, c)  # [n, 256, c]
                encoded = x  # Update encoded output

        ################# Pooling ##############

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, use_bias=False, name="head", **kw) # Gemma3 doesn't use bias
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        if is_multicrop:
            # Reshape back: [B*N, patches, D] → [B, N*patches, D]
            n_patches = x.shape[1] if hasattr(x, 'shape') else encoded.shape[1]
            encoded = encoded.reshape(B, N * n_patches, -1)
            if hasattr(x, 'shape'):
                x = x.reshape(B, N * n_patches, -1)

        return x, out """

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # ADD: Handle Pan & Scan multi-crop input
        if len(image.shape) == 5:  # [B, N_crops, H, W, C]
            B, N, H, W, C = image.shape
            image = image.reshape(B * N, H, W, C)  # Flatten crops
            is_multicrop = True
        else:
            is_multicrop = False
            B = image.shape[0]

        *batch_dims, _, _, _ = image.shape
        image = einops.rearrange(image, "... h w c -> (...) h w c")

        image = jnp.asarray(image, jnp.float32)

        # Patch extraction via Conv - automatically handles any image size (224×224 or 896×896)
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])  # x.shape is now [N, H*W, C]. For 896/14=64, H*W=4096.

        # Store H, W for final reshaping if needed
        H_orig, W_orig = h, w

        seqshape = self.posemb_shape or (h, w)
        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, seqshape, c, "pos_embedding", jnp.float32)

        # if self.pool_type == "tok":
        #     cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
        #     x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # ====================================================================
        # CORRECT LOCATION FOR THE SOFTNORM LAYER
        # ====================================================================
        # The layer name "mm_soft_embedding_norm" will cause Flax to create
        # a parameter path that matches your checkpoint.
        x, _ = CommonRMSNorm(name="mm_soft_embedding_norm")(x, None)  # Gemma3 has no bias
        # ====================================================================

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        # ====================================================================
        # GEMMA3 POOLING LOGIC: Reduce to 256 tokens (16x16 grid)
        # ====================================================================
        # Supports different input resolutions:
        # - 896x896 with 14x14 patches → 64x64 grid (4096 patches) → 4x4 pool → 16x16 (256 tokens)
        # - 224x224 with 14x14 patches → 16x16 grid (256 patches) → no pooling needed
        if self.gemma3_pooling:
            n, seq_len, c = encoded.shape
            grid_size = int(seq_len**0.5)
            assert grid_size * grid_size == seq_len, f"Sequence length {seq_len} must be perfect square"

            if grid_size == 64:
                # 896x896 input: Pool 64×64 → 16×16 using 4×4 avg_pool
                x_2d = encoded.reshape(n, grid_size, grid_size, c)  # [n, 64, 64, c]
                x_2d_pooled = nn.avg_pool(x_2d, window_shape=(4, 4), strides=(4, 4))
                encoded = x_2d_pooled.reshape(n, 16 * 16, c)  # [n, 256, c]
            elif grid_size == 16:
                # 224x224 input: Already 16×16 = 256 tokens, no pooling needed
                pass
            elif grid_size == 32:
                # 448x448 input: Pool 32×32 → 16×16 using 2×2 avg_pool
                x_2d = encoded.reshape(n, grid_size, grid_size, c)  # [n, 32, 32, c]
                x_2d_pooled = nn.avg_pool(x_2d, window_shape=(2, 2), strides=(2, 2))
                encoded = x_2d_pooled.reshape(n, 16 * 16, c)  # [n, 256, c]
            else:
                raise ValueError(
                    f"Unsupported grid size {grid_size}x{grid_size} for gemma3_pooling. "
                    f"Expected 64 (896x896), 32 (448x448), or 16 (224x224) with 14x14 patches."
                )

        # NOTE: 'x' is now only used as a reference to 'encoded' for the legacy pooling/head paths
        x = encoded
        # ====================================================================

        ################# Legacy Pooling (Skip if pool_type="none") ##############

        """ if self.pool_type == "map":
            raise ValueError(f"map not supported for pooling type: '{self.pool_type}'")
            # x = out["head_input"] = MAPHead(
            #     num_heads=self.num_heads,
            #     mlp_dim=self.mlp_dim,
            #     dtype=self.dtype_mm,
            # )(x)
        elif self.pool_type == "gap":
            #x = out["head_input"] = jnp.mean(x, axis=1) # [N, C]
            raise ValueError(f"gap not supported for pooling type: '{self.pool_type}'")
        elif self.pool_type == "0" or self.pool_type == "tok":
            x = out["head_input"] = x[:, 0] # [N, C]
            # Need to adjust encoded for tok pooling if not already handled
            if self.pool_type == "tok":
                 encoded = encoded[:, 1:]
            raise ValueError(f"0 or tok supported for pooling type: '{self.pool_type}'")
        elif self.pool_type == "none":
            # If "none", we are returning the sequence of tokens (now 256 tokens)

            x_2d = jnp.reshape(encoded, [n, 16, 16, -1]) # Reshape 256 tokens to 16x16 grid
            
            # Set x and x_2d to the pooled sequence/grid for the final output section
            x = encoded # [N, 256, C]
            
            out["pre_logits_2d"] = x_2d
            out["pre_logits"] = x

            #The key fix: Skip the classification head entirely when pool_type="none"
            #return x, out 
            #pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'") """

        # NOTE: This model is now dedicated to outputting the 256 token sequence.

        if self.pool_type != "none":
            # If a pool type other than "none" is used, it indicates a misconfiguration
            # for the intended multi-token output task.
            raise ValueError(
                f"Only 'none' pooling is supported for the 256-token output path. Found: '{self.pool_type}'"
            )

        # Since pool_type is "none", we proceed with the 256 tokens sequence (x=encoded).

        # Reshape for 2D representation (16x16 grid)
        x_2d = jnp.reshape(encoded, [n, 16, 16, c])

        # Set x and x_2d to the pooled sequence/grid for the final output section
        x = encoded  # [N, 256, C]

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        # Now, do embedding projection (named "head" to match checkpoint)
        kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
        head = nn.Dense(self.num_classes, dtype=self.dtype_mm, use_bias=False, name="head", **kw)
        x_2d = out["logits_2d"] = head(x_2d)
        x = out["logits"] = head(x)

        return x, out

        # The code below is for when pool_type is NOT "none" (i.e., classification)

        # Reshape for 2D logits (use the original image grid H_orig, W_orig if no pooling was done,
        # or the pooled grid 16x16 if gemma3_pooling was active and pool_type was NOT "none")
        # if self.gemma3_pooling:
        #     x_2d = jnp.reshape(encoded, [n, 16, 16, -1])
        # else:
        #     x_2d = jnp.reshape(encoded, [n, H_orig, W_orig, -1])

        # if self.rep_size:
        #     rep_size = self.width if self.rep_size is True else self.rep_size
        #     hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
        #     x_2d = nn.tanh(hid(x_2d))
        #     x = nn.tanh(hid(x))

        # out["pre_logits_2d"] = x_2d
        # out["pre_logits"] = x

        # # This section is skipped by the 'return x, out' above when pool_type="none"
        # if self.num_classes:
        #     kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
        #     head = nn.Dense(self.num_classes, dtype=self.dtype_mm, use_bias=False, name="mm_input_projection", **kw)
        #     x_2d = out["logits_2d"] = head(x_2d)
        #     x = out["logits"] = head(x)

        # if is_multicrop:
        #     # Reshape back: [B*N, patches, D] → [B, N*patches, D]
        #     n_patches = x.shape[1] if hasattr(x, 'shape') else encoded.shape[1]
        #     encoded = encoded.reshape(B, N * n_patches, -1)
        #     if hasattr(x, 'shape'):
        #         x = x.reshape(B, N * n_patches, -1)

        # # Final return when classification is run (not used by LAP)
        # print(f"DEBUG siglip return: x.shape={x.shape}, out.keys={list(out.keys())}")
        # return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
