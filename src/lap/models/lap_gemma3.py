"""Gemma3-specific LAP implementation.

This module contains the LAPGemma3 class which inherits from LAP and overrides
only the Gemma3-specific behavior:
- Image placeholder token replacement in the tokenized sequence (vs concatenation in LAP)
- Different EOS token (106 vs 1) and vocab size (262144 vs 257152)
- Gemma3 LLM backbone with QK-normalization (vs softcapping in Gemma2)
- SigLIP Gemma3 vision encoder with configurable resolution and automatic pooling to 256 tokens
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import openpi.models.model as _model
from openpi.models.model import Observation
from openpi.shared import array_typing as at
from typing_extensions import override

import lap.models.backbones.gemma3 as _gemma3
from lap.models.backbones.gemma3 import get_config as get_gemma3_config
import lap.models.backbones.siglip_gemma3 as _siglip_gemma3
from lap.models.lap import LAP
import lap.models.lap_config as _lap_config
from lap.models.model_adapter import CoTObservation

logger = logging.getLogger("openpi")

# Gemma3 constants
GEMMA3_VOCAB_SIZE = 262_144
GEMMA3_IMAGE_TOKEN = 262145  # Placeholder token ID for image embeddings (out-of-vocab)
GEMMA3_EOS_TOKEN = 1  # Actual EOS token (was incorrectly 106 which is <start_of_turn>)
GEMMA3_BEGIN_IMAGE_TOKEN = 255999
GEMMA3_END_IMAGE_TOKEN = 262144

# Gemma3 SigLIP constants
SIGLIP_PATCH_SIZE = 14  # 14x14 pixel patches
SIGLIP_NUM_PATCHES = 256  # Output tokens after pooling (16x16 grid)


@jax.vmap
def _left_to_right_align_with_image_mask(x, input_mask, attn_mask, image_mask):
    """Converts input from left-align to right-aligned, including image_mask.

    Extension of pi0_fast.left_to_right_align that also handles image_mask.
    """
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert image_mask.ndim == 1
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    assert image_mask.shape[0] == input_mask.shape[0]

    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    image_mask = jnp.roll(image_mask, -seqlen, axis=0)
    return x, input_mask, attn_mask, image_mask


class LAPGemma3(LAP):
    """Gemma3-specific LAP implementation.

    Inherits from LAP and overrides only Gemma3-specific behavior:
    - Image placeholders in tokenized sequence are replaced with actual embeddings
      (vs concatenating [image_tokens, text_tokens] in LAP)
    - Different EOS token (106 instead of 1) and vocab size (262144 instead of 257152)
    - Gemma3 LLM backbone with QK-normalization (vs softcapping in Gemma2)
    - SigLIP Gemma3 encoder with automatic pooling to 256 tokens per image
    """

    # Gemma3-specific token constants
    EOS_TOKEN = GEMMA3_EOS_TOKEN
    VOCAB_SIZE = GEMMA3_VOCAB_SIZE
    IMAGE_TOKEN = GEMMA3_IMAGE_TOKEN
    BEGIN_IMAGE_TOKEN = GEMMA3_BEGIN_IMAGE_TOKEN
    END_IMAGE_TOKEN = GEMMA3_END_IMAGE_TOKEN

    def __init__(self, config: _lap_config.LAPConfig, rngs: nnx.Rngs):
        """Initialize Gemma3-specific model components.

        Unlike LAP which uses PaliGemma's SigLIP, Gemma3 uses:
        - SigLIP Gemma3 encoder with automatic pooling to 256 tokens
        - Gemma3 LLM backbone with QK-normalization
        - Image resolution is configurable (default 224x224, supports up to 896x896)
        """
        # Initialize base model (skip LAP's __init__ which loads wrong backbone)
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)

        # Validate config
        assert "gemma3" in config.paligemma_variant, "LAPGemma3 requires gemma3 variant"
        assert "gemma3" in config.action_expert_variant, "gemma3 must be used for both LLM and action expert"

        # Shared training and loss attributes from LAP.
        self._configure_shared_training_attributes(config)
        self.image_resolution = config.image_resolution
        self.tokenizer = None  # Can be set externally for decoding

        # Get Gemma3 configs
        paligemma_config = get_gemma3_config(config.paligemma_variant)
        action_expert_config = get_gemma3_config(config.action_expert_variant)

        # Initialize Gemma3 LLM with both VLM and action expert
        llm = nnx_bridge.ToNNX(
            _gemma3.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                stop_action_to_vlm_grad=config.stop_action_to_vlm_grad,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # Compute positional embedding grid size based on image resolution
        # grid_size = image_resolution / patch_size (14)
        img_h, img_w = self.image_resolution
        posemb_h = img_h // SIGLIP_PATCH_SIZE  # e.g., 896/14=64 or 224/14=16
        posemb_w = img_w // SIGLIP_PATCH_SIZE

        # Initialize Gemma3-specific SigLIP with configurable resolution
        img = nnx_bridge.ToNNX(
            _siglip_gemma3.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",  # Keep 256 tokens after gemma3_pooling
                gemma3_pooling=True,
                scan=True,
                dtype_mm=config.dtype,
                posemb="learn",
                posemb_shape=(posemb_h, posemb_w),  # Dynamic based on image_resolution
            )
        )
        # Initialize with correctly sized fake image
        fake_obs = config.fake_obs()
        fake_obs_image = next(iter(fake_obs.images.values()))
        b, h, w, c = fake_obs_image.shape
        new_shape = (b, 1, img_h, img_w, c)
        fake_image_resized = jax.image.resize(fake_obs_image[:, None], new_shape, method="linear")
        img.lazy_init(fake_image_resized, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Initialize action expert projections (same as LAP)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    # ============ Gemma3-Specific: Image Embedding ============

    def _embed_images(
        self,
        obs: CoTObservation | Observation,
    ) -> tuple[at.Float[at.Array, "b n_patches emb"], at.Bool[at.Array, "b n_patches"]]:
        """Embed images using Gemma3-specific SigLIP encoder.

        SigLIP Gemma3 handles 4D input [b, h, w, c] directly (no need for time dimension).
        After pooling, outputs 256 tokens per image regardless of input resolution.

        Returns:
            (image_embeddings, image_mask) where:
            - image_embeddings: [b, total_patches, emb] concatenated across all images
            - image_mask: [b, total_patches] mask for valid image positions
        """
        all_tokens = []
        all_masks = []

        for name in obs.images:
            image = obs.images[name]
            # SigLIP Gemma3 accepts [b, h, w, c] directly - no time dimension needed
            image_tokens, _ = self.PaliGemma.img(image, train=False)
            num_patches = image_tokens.shape[1]

            all_tokens.append(image_tokens)
            all_masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=num_patches))

        image_embeddings = jnp.concatenate(all_tokens, axis=1)
        image_mask = jnp.concatenate(all_masks, axis=1)

        return image_embeddings, image_mask

    # ============ Gemma3-Specific: Placeholder Replacement ============

    def _replace_placeholders(
        self,
        token_embeddings: at.Float[at.Array, "b s emb"],
        token_mask: at.Bool[at.Array, "b s"],
        token_ar_mask: at.Bool[at.Array, "b s"],
        tokenized_sequence: at.Int[at.Array, "b s"],
        image_embeddings: at.Float[at.Array, "b n_patches emb"],
        image_mask: at.Bool[at.Array, "b n_patches"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Replace IMAGE_TOKEN placeholders with actual image embeddings and masks.

        The tokenized sequence contains IMAGE_TOKEN (262145) at positions where
        image embeddings should be inserted. This method:
        1. Finds all placeholder positions
        2. Replaces placeholder embeddings with corresponding image embeddings
        3. Updates masks to use actual image masks at placeholder positions

        Args:
            token_embeddings: Text embeddings (placeholders have dummy embeddings)
            token_mask: Attention mask for tokens
            token_ar_mask: Autoregressive mask for tokens
            tokenized_sequence: Token IDs with IMAGE_TOKEN placeholders
            image_embeddings: Actual image embeddings from SigLIP
            image_mask: Mask for valid image patches

        Returns:
            (embeddings, mask, ar_mask) with placeholders replaced
        """
        is_placeholder = tokenized_sequence == self.IMAGE_TOKEN
        b, s, emb = token_embeddings.shape
        _, n_patches = image_mask.shape

        # Map each placeholder to its corresponding image embedding index
        # cumsum gives 1-indexed position, subtract 1 for 0-indexed
        placeholder_idx = jnp.cumsum(is_placeholder, axis=1) - 1
        placeholder_idx = jnp.clip(placeholder_idx, 0, n_patches - 1)

        # Gather image embeddings and masks at placeholder positions
        batch_idx = jnp.arange(b)[:, None]
        selected_img_embs = image_embeddings[batch_idx, placeholder_idx]
        selected_img_mask = image_mask[batch_idx, placeholder_idx]

        # Replace: use image values where is_placeholder, else original values
        result_embeddings = jnp.where(is_placeholder[..., None], selected_img_embs, token_embeddings)
        result_mask = jnp.where(is_placeholder, selected_img_mask, token_mask)
        # Image tokens are non-autoregressive (can attend to each other)
        result_ar_mask = jnp.where(is_placeholder, False, token_ar_mask)

        return result_embeddings, result_mask, result_ar_mask

    # ============ Override: embed_prefix ============

    @override
    def embed_prefix(
        self,
        obs: CoTObservation | Observation,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        """Embed images and text for the prefix using Gemma3's placeholder approach.

        Unlike LAP which concatenates [image_tokens, text_tokens], Gemma3 uses
        placeholder tokens in the text sequence that get replaced with image embeddings.
        This keeps the sequence structure matching the tokenizer's format.

        Returns:
            (prefix_tokens, prefix_mask, prefix_ar_mask, image_mask) where:
            - prefix_tokens: embeddings with placeholders replaced by image embeddings
            - prefix_mask: attention mask for valid positions
            - prefix_ar_mask: autoregressive mask (True for causal, False for bidirectional)
            - image_mask: boolean mask indicating which positions are image tokens
        """
        # Step 1: Embed the tokenized sequence (placeholders get dummy embeddings)
        # Replace IMAGE_TOKEN (262145, out-of-vocab) with 0 to avoid indexing errors
        tokenized_safe = jnp.where(
            obs.tokenized_prompt == self.IMAGE_TOKEN,
            0,  # Temporary placeholder - will be replaced below
            obs.tokenized_prompt,
        )
        token_embeddings = self.PaliGemma.llm(tokenized_safe, method="embed")

        # Get text masks
        token_mask = obs.tokenized_prompt_mask
        # Attention pattern for text tokens:
        # - Prompt tokens (tokenized_langact_mask=False): ar_mask=False → bidirectional
        # - Langact tokens (tokenized_langact_mask=True): ar_mask=True → causal
        # - Image placeholders will be set to ar_mask=False in _replace_placeholders
        if obs.tokenized_langact_mask is not None:
            token_ar_mask = obs.tokenized_langact_mask
        else:
            token_ar_mask = jnp.zeros_like(token_mask, dtype=bool)

        # Step 2: Get image embeddings from SigLIP
        image_embeddings, siglip_image_mask = self._embed_images(obs)

        # Step 3: Replace placeholders with actual image embeddings and masks
        prefix_tokens, prefix_mask, prefix_ar_mask = self._replace_placeholders(
            token_embeddings,
            token_mask,
            token_ar_mask,
            obs.tokenized_prompt,
            image_embeddings,
            siglip_image_mask,
        )

        # Compute image_mask: positions where image tokens are (for bidirectional attention)
        image_mask = obs.tokenized_prompt == self.IMAGE_TOKEN

        return prefix_tokens, prefix_mask, prefix_ar_mask, image_mask

    # ============ Override: _build_prefix_action_mask ============

    @override
    def _build_prefix_action_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b s"],
        observation: CoTObservation | Observation,
    ) -> at.Bool[at.Array, "b s"]:
        """Build prefix mask for action attention, excluding langact tokens.

        Action tokens should attend to images + prompt, but NOT langact tokens.
        For Gemma3, images are embedded inline (via placeholders) so prefix length
        equals tokenized_prompt length - no separate image prefix adjustment needed.
        """
        if observation.tokenized_langact_mask is None:
            return prefix_mask
        # Return True for images + prompt (non-langact), False for langact
        return jnp.logical_and(prefix_mask, jnp.logical_not(observation.tokenized_langact_mask))

    # ============ Override: compute_loss hook ============

    @override
    def _embed_prefix_for_loss(
        self,
        observation: CoTObservation | Observation,
        suffix_inputs: dict[str, at.Array] | None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        dict[str, at.Array],
    ]:
        """Attach Gemma3 image_mask to the shared LAP compute_loss forward call."""
        prefix_tokens, prefix_mask, prefix_ar_mask, image_mask = self.embed_prefix(observation)
        if suffix_inputs is not None:
            suffix_len = suffix_inputs["suffix_tokens"].shape[1]
            image_mask = jnp.concatenate(
                [image_mask, jnp.zeros((image_mask.shape[0], suffix_len), dtype=bool)],
                axis=1,
            )
        else:
            image_mask = image_mask.astype(bool)
        return prefix_tokens, prefix_mask, prefix_ar_mask, {"image_mask": image_mask}

    # ============ Override: sample_actions ============

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions with Gemma3's image_mask for bidirectional image attention."""
        import openpi.models.pi0 as _pi0

        from lap.models.model_adapter import preprocess_observation

        observation = preprocess_observation(
            None, observation, train=False, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Embed prefix with image_mask (Gemma3 returns 4 values)
        prefix_tokens, prefix_mask, prefix_ar_mask, image_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Fill KV cache with prefix, passing image_mask
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            image_mask=image_mask,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            gemma_out, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                # Note: image_mask not needed here as KV cache already has image info
            )
            suffix_out = gemma_out[1]
            assert gemma_out[0] is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ============ Override: sample_tokens ============

    @override
    def sample_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 390,
        temperature: float = 0.0,
    ) -> _model.Actions:
        """Sample tokens with Gemma3's image_mask for bidirectional image attention."""
        import openpi.models.pi0 as _pi0
        import openpi.models.pi0_fast as _pi0_fast

        from lap.models.model_adapter import preprocess_observation

        observation = preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
            aug_wrist_image=self.aug_wrist_image,
        )

        # Embed prefix with image_mask (Gemma3 returns 4 values)
        prefix_token_embeddings, prefix_mask, prefix_ar_mask, image_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)

        # Right-align sequences (including image_mask)
        prefix_token_embeddings, prefix_mask, prefix_attn_mask, image_mask = _left_to_right_align_with_image_mask(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask, image_mask
        )

        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # Fill KV cache with prefix, passing image_mask
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        pre_logits, kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None] if self.enable_action_training else [prefix_token_embeddings],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None] if self.enable_action_training else [None],
            image_mask=image_mask,
        )

        # Prepare decoding
        last_logit = self.PaliGemma.llm(pre_logits[0][:, -1:], method="decode")
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def step(carry):
            rng, last_logit, output_tokens, cache, eos_mask, step = carry

            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1).astype(jnp.int32),
                lambda _: jnp.argmax(last_logit, axis=-1).astype(jnp.int32),
                operand=None,
            )
            output_tokens = _pi0_fast.put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )

            eos_token = jnp.squeeze(token, axis=-1)
            eos_mask = eos_mask | (eos_token == self.EOS_TOKEN)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_prelogit, kv_cache = self.PaliGemma.llm(
                [token_embedding, None] if self.enable_action_training else [token_embedding],
                mask=mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, None] if self.enable_action_training else [None],
                # Note: image_mask not needed here as KV cache already has image info
            )
            last_logit = self.PaliGemma.llm(last_prelogit[0], method="decode")

            return rng, last_logit, output_tokens, kv_cache, eos_mask, step + 1

        def cond(carry):
            _, _, _, _, eos_mask, step = carry
            return (~jnp.all(eos_mask)) & (step < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, jnp.zeros((last_logit.shape[0],), dtype=bool), 0)
        )
        return output_tokens
