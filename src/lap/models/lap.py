import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import openpi.models.model as _model
from openpi.models.model import Observation
import openpi.models.pi0 as _pi0
import openpi.models.pi0_fast as _pi0_fast
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from typing_extensions import override

from lap.datasets.registry import VQA_DATASET_ID_MAP
import lap.models.backbones.gemma as _gemma
import lap.models.lap_config as _lap_config
from lap.models.model_adapter import CoTObservation
from lap.models.model_adapter import preprocess_observation
from lap.models.model_utils.metrics import compute_per_vqa_dataset_metrics
from lap.models.model_utils.metrics import compute_sample_specific_metrics
from lap.models.model_utils.metrics import compute_token_accuracy_metrics
from lap.models.model_utils.visualization import log_attention_mask_wandb as _log_attention_mask_wandb

logger = logging.getLogger("openpi")
PALIGEMMA_VOCAB_SIZE = 257_152
log_attention_mask_wandb = _log_attention_mask_wandb


class LAP(_pi0.Pi0):
    EOS_TOKEN: int = 1
    VOCAB_SIZE: int = PALIGEMMA_VOCAB_SIZE

    def __init__(self, config: _lap_config.LAPConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self._configure_shared_training_attributes(config)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        if self.enable_action_training:
            action_expert_config = _gemma.get_config(config.action_expert_variant)
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    stop_action_to_vlm_grad=config.stop_action_to_vlm_grad,
                    cache_dtype=config.dtype,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
            self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            if config.pi05:
                self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            else:
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                self.action_time_mlp_in = nnx.Linear(
                    2 * action_expert_config.width, action_expert_config.width, rngs=rngs
                )
                self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        else:
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    stop_action_to_vlm_grad=False,
                    cache_dtype=config.dtype,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _configure_shared_training_attributes(self, config: _lap_config.LAPConfig) -> None:
        self.pi05 = config.pi05
        self.verbose_mode = config.verbose_mode
        self.aug_wrist_image = config.aug_wrist_image
        self.enable_image_augmentation = config.enable_image_augmentation
        self.image_keys = config.image_keys
        self.enable_action_training = bool(config.enable_action_training)
        self.enable_langact_training = bool(config.enable_langact_training)
        self.enable_prediction_training = bool(config.enable_prediction_training)
        self.enable_vqa_training = bool(config.enable_vqa_training)
        self.language_loss_weight = float(getattr(config, "language_loss_weight", 1.0))
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.prediction_loss_weight = float(getattr(config, "prediction_loss_weight", 0.2))
        self.vqa_loss_weight = float(getattr(config, "vqa_loss_weight", 0.1))
        vqa_loss_weights_dict = getattr(config, "vqa_loss_weights", None)
        if vqa_loss_weights_dict is not None:
            self.vqa_loss_weights_by_id = {
                VQA_DATASET_ID_MAP[name]: weight
                for name, weight in vqa_loss_weights_dict.items()
                if name in VQA_DATASET_ID_MAP
            }
        else:
            self.vqa_loss_weights_by_id = None

    # TODO: make all obs.images into a single forward pass
    def embed_prefix(
        self, obs: CoTObservation | Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
    ]:
        tokens = []
        input_mask = []
        ar_mask = []

        for name in obs.images:
            image = obs.images[name]
            image_tokens, _ = self.PaliGemma.img(image, train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # All image tokens attend to each other (no autoregressive masking)
            ar_mask.append(
                einops.repeat(
                    jnp.array([False] * image_tokens.shape[0]),
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )

        tokens.append(self.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
        input_mask.append(obs.tokenized_prompt_mask)

        # Attention pattern for text tokens:
        # - Prompt tokens (tokenized_langact_mask=False): ar_mask=False → bidirectional
        # - Langact tokens (tokenized_langact_mask=True): ar_mask=True → causal
        # This allows:
        #   - Images: bidirectional (ar_mask=False)
        #   - Prompt: bidirectional, can attend to images
        #   - Langact: causal, can attend to images + all prompt
        if obs.tokenized_langact_mask is not None:
            ar_mask.append(obs.tokenized_langact_mask)
        else:
            text_ar_mask = jnp.zeros((obs.tokenized_prompt.shape[0], obs.tokenized_prompt.shape[1]), dtype=bool)
            ar_mask.append(text_ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)

        return tokens, input_mask, ar_mask

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
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        return prefix_tokens, prefix_mask, prefix_ar_mask, {}

    def prepare_suffix(
        self,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        noise_rng: at.KeyArrayLike,
        time_rng: at.KeyArrayLike,
    ) -> dict[str, at.Array]:
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_ar_mask = einops.repeat(suffix_ar_mask, "s -> b s", b=suffix_tokens.shape[0])
        return {
            "suffix_tokens": suffix_tokens,
            "suffix_mask": suffix_mask,
            "suffix_ar_mask": suffix_ar_mask,
            "adarms_cond": adarms_cond,
            "u_t": u_t,
        }

    def _compute_language_loss(
        self,
        observation: CoTObservation | Observation,
        prefix_pre_logits: at.Float[at.Array, "b s emb"],
        sample_mask: at.Bool[at.Array, "b"] | None = None,
        *,
        verbose_mode: bool = False,
        return_predictions: bool = False,
        loss_name: str = "lang_loss",
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute language/reasoning loss given precomputed prefix pre-logits."""

        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.VOCAB_SIZE,
        )

        # Drop final prefix token (no next-token target) then align to text targets.
        pre_logits = prefix_pre_logits[:, :-1]
        pre_logits = pre_logits[:, -targets.shape[1] :]
        logits = self.PaliGemma.llm(pre_logits, method="decode")

        loss_mask = jnp.logical_and(
            observation.tokenized_langact_mask[:, 1:],
            jnp.logical_and(observation.tokenized_prompt_mask[:, 1:], observation.token_loss_mask[:, 1:]),
        )
        if sample_mask is not None:
            ex_mask = jnp.asarray(sample_mask)[..., None]
            loss_mask = loss_mask * ex_mask
        else:
            ex_mask = None

        def prepare_mask(mask):
            if mask is None:
                return None
            shifted_mask = mask[:, 1:]
            if ex_mask is not None:
                return shifted_mask * ex_mask
            return shifted_mask

        if verbose_mode:
            critical_mask = prepare_mask(observation.critical_token_mask)
            number_mask = prepare_mask(observation.number_token_mask)
            direction_mask = prepare_mask(observation.direction_token_mask)
        else:
            critical_mask = number_mask = direction_mask = None

        logp = jax.nn.log_softmax(logits, axis=-1)
        token_pplx = jnp.sum(targets * logp, axis=-1)
        # Standard hard target loss
        per_sample_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
        metrics = {loss_name: jnp.mean(per_sample_loss)}

        # Return predictions if requested (independently of verbose_mode)
        # IMPORTANT: Predictions are ONLY returned when return_predictions=True
        if return_predictions:
            predictions = jnp.argmax(logits, axis=-1)
            metrics["predictions"] = predictions
            metrics["labels"] = observation.tokenized_prompt[:, 1:]
            metrics["token_mask"] = loss_mask

        # Compute detailed accuracy metrics if verbose_mode is enabled
        # NOTE: When verbose_mode=True but return_predictions=False, predictions are
        # computed internally for accuracy metrics but NOT added to the output metrics
        if verbose_mode:
            # Reuse predictions if already computed (when return_predictions=True),
            # otherwise compute them temporarily for accuracy calculation only
            predictions = metrics.get("predictions", jnp.argmax(logits, axis=-1))
            accuracy_metrics = compute_token_accuracy_metrics(
                predictions=predictions,
                labels=observation.tokenized_prompt[:, 1:],
                per_token_loss=-token_pplx * loss_mask,
                token_mask=loss_mask,
                critical_mask=critical_mask,
                number_mask=number_mask,
                direction_mask=direction_mask,
            )
            # Only add accuracy metrics, not predictions (unless already added above)
            metrics.update(accuracy_metrics)

        return per_sample_loss, metrics

    def _compute_action_loss(
        self,
        suffix_out: at.Float[at.Array, "b s emb"],
        u_t: at.Float[at.Array, "b ah ad"],
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        """Compute action diffusion loss from suffix activations."""

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per_sample_action_loss = jnp.mean(jnp.square(v_t - u_t), axis=(-1, -2))
        metrics = {"action_loss": jnp.mean(per_sample_action_loss)}
        return per_sample_action_loss, metrics

    def _build_prefix_action_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b s"],
        observation: CoTObservation | Observation,
    ) -> at.Bool[at.Array, "b s"]:
        """Build prefix mask for action attention, excluding langact tokens.

        Action tokens should attend to images + prompt, but NOT langact tokens.
        This creates a mask that is True for images and prompt, False for langact.
        """
        if observation.tokenized_langact_mask is None:
            return prefix_mask
        # Expand langact_mask to full prefix length (add zeros for image positions)
        img_seq_len = prefix_mask.shape[1] - observation.tokenized_langact_mask.shape[1]
        langact_mask_full = jnp.concatenate(
            [
                jnp.zeros((observation.tokenized_langact_mask.shape[0], img_seq_len), dtype=bool),
                observation.tokenized_langact_mask,
            ],
            axis=1,
        )
        # Return True for images + prompt (non-langact), False for langact
        return jnp.logical_and(prefix_mask, jnp.logical_not(langact_mask_full))

    def _build_combined_attention_mask(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
        suffix_ar_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Bool[at.Array, "b _t _s"]:
        """Build combined attention mask for prefix (VLM) and suffix (action expert).

        Current attention pattern (using tokenized_langact_mask as ar_mask):
        - Image tokens: bidirectional (ar_mask=False)
        - Prompt tokens: bidirectional (ar_mask=False from tokenized_langact_mask)
        - Langact tokens: causal (ar_mask=True from tokenized_langact_mask)
        - Action tokens: causal, can attend to images + prompt, NOT langact

        This means images and prompt can attend to each other bidirectionally,
        while langact is causal and can attend to all images + prompt.
        Action tokens use prefix_mask_action which EXCLUDES langact.
        """
        prefix_attn = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        if suffix_mask is None or suffix_ar_mask is None:
            return prefix_attn

        batch_size, prefix_len = prefix_mask.shape
        suffix_len = suffix_mask.shape[1]
        combined = jnp.zeros((batch_size, prefix_len + suffix_len, prefix_len + suffix_len), dtype=bool)
        combined = combined.at[:, :prefix_len, :prefix_len].set(prefix_attn)

        # For action rows: ar_mask=False for prefix (full attention to img+prompt),
        # ar_mask=True for suffix (causal among action tokens)
        # prefix_mask_action excludes langact so action cannot attend to langact
        prefix_ar_mask_action = jnp.zeros_like(prefix_mask_action, dtype=bool)
        input_mask = jnp.concatenate([prefix_mask_action, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask_action, suffix_ar_mask], axis=1)
        action_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        combined = combined.at[:, prefix_len:, :].set(action_mask[:, prefix_len:, :])
        return combined

    def _build_combined_positions(
        self,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_mask_action: at.Bool[at.Array, "b p"],
        suffix_mask: at.Bool[at.Array, "b s"] | None,
    ) -> at.Int[at.Array, "b _t"]:
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        if suffix_mask is None:
            return prefix_positions.astype(jnp.int32)
        suffix_positions = jnp.sum(prefix_mask_action, axis=-1, keepdims=True) + jnp.cumsum(suffix_mask, axis=-1) - 1
        combined = jnp.concatenate([prefix_positions, suffix_positions], axis=1)
        return combined.astype(jnp.int32)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        stage_config: dict | None = None,
        verbose_mode: bool | None = None,
        return_augmented_images: bool = False,
    ) -> dict[str, at.Array]:
        preprocess_rng, _, noise_rng, time_rng = jax.random.split(rng, 4)

        # Use passed verbose_mode if provided, otherwise use class attribute
        effective_verbose_mode = verbose_mode if verbose_mode is not None else self.verbose_mode

        # Determine batch size
        batch_size = observation.tokenized_prompt.shape[0]

        # Compute VQA mask first (before preprocessing) to skip augmentation for VQA samples
        vqa_mask = None
        if self.enable_vqa_training and hasattr(observation, "is_vqa_sample") and observation.is_vqa_sample is not None:
            vqa_mask = jnp.asarray(observation.is_vqa_sample, dtype=bool)
        pred_mask = None
        if (
            self.enable_prediction_training
            and hasattr(observation, "is_prediction_sample")
            and observation.is_prediction_sample is not None
        ):
            pred_mask = jnp.asarray(observation.is_prediction_sample, dtype=bool)

        # Preprocess observation (will skip augmentation for VQA samples if vqa_mask is provided)
        observation = preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
            aug_wrist_image=self.aug_wrist_image,
            enable_image_augmentation=self.enable_image_augmentation,
            vqa_mask=vqa_mask,
        )

        # Store augmented images for later visualization (if requested)
        augmented_images = observation.images if return_augmented_images else None

        suffix_inputs = (
            self.prepare_suffix(observation, actions, noise_rng, time_rng) if self.enable_action_training else None
        )
        # Build prefix for langact/action losses (first frame + text by default).
        # Subclasses can override to attach additional forward kwargs (e.g., image_mask).
        prefix_tokens, prefix_mask, prefix_ar_mask, model_forward_kwargs = self._embed_prefix_for_loss(
            observation, suffix_inputs
        )

        prefix_mask_action = (
            self._build_prefix_action_mask(prefix_mask, observation) if self.enable_action_training else prefix_mask
        )
        combined_mask = self._build_combined_attention_mask(
            prefix_mask,
            prefix_ar_mask,
            prefix_mask_action,
            suffix_inputs["suffix_mask"] if self.enable_action_training else None,
            suffix_inputs["suffix_ar_mask"] if self.enable_action_training else None,
        )
        combined_positions = self._build_combined_positions(
            prefix_mask, prefix_mask_action, suffix_inputs["suffix_mask"] if self.enable_action_training else None
        )

        pre_logits, _ = self.PaliGemma.llm(
            embedded=[prefix_tokens, suffix_inputs["suffix_tokens"]]
            if self.enable_action_training
            else [prefix_tokens],
            positions=combined_positions,
            mask=combined_mask,
            adarms_cond=[None, suffix_inputs["adarms_cond"]] if self.enable_action_training else [None],
            **model_forward_kwargs,
        )

        metrics = {}
        lang_per_sample_loss = jnp.zeros(batch_size, dtype=jnp.float32)
        action_per_sample_loss = jnp.zeros(batch_size, dtype=jnp.float32)

        if self.enable_langact_training:
            combined_langact_mask = observation.sample_mask
            lang_loss, lang_metrics = self._compute_language_loss(
                observation,
                pre_logits[0],
                sample_mask=combined_langact_mask,
                verbose_mode=effective_verbose_mode,
            )
            metrics.update(lang_metrics)

            if self.enable_vqa_training or self.enable_prediction_training:
                if vqa_mask is None:
                    vqa_mask = jnp.zeros(batch_size, dtype=bool)
                if pred_mask is None:
                    pred_mask = jnp.zeros(batch_size, dtype=bool)
                lang_mask = jnp.logical_not(jnp.logical_or(vqa_mask, pred_mask))

                vqa_mask = jnp.logical_and(vqa_mask, combined_langact_mask)
                pred_mask = jnp.logical_and(pred_mask, combined_langact_mask)
                lang_mask = jnp.logical_and(lang_mask, combined_langact_mask)
                num_active_samples = jnp.maximum(jnp.sum(combined_langact_mask), 1.0)
                metrics["active_num_samples"] = jnp.sum(combined_langact_mask)
                metrics["active_sample_portion"] = metrics["active_num_samples"] / jnp.maximum(batch_size, 1.0)
                metrics["vqa_num_samples"] = jnp.sum(vqa_mask)
                metrics["vqa_sample_portion"] = metrics["vqa_num_samples"] / num_active_samples
                metrics["pred_num_samples"] = jnp.sum(pred_mask)
                metrics["pred_sample_portion"] = metrics["pred_num_samples"] / num_active_samples
                metrics["langact_num_samples"] = jnp.sum(lang_mask)
                metrics["langact_sample_portion"] = metrics["langact_num_samples"] / num_active_samples

                if self.enable_vqa_training:
                    metrics.update(
                        compute_sample_specific_metrics(
                            per_sample_loss=lang_loss,
                            sample_mask=vqa_mask,
                            prefix="vqa_",
                        )
                    )
                    # Add per-VQA-dataset metrics if vqa_dataset_id is available
                    if hasattr(observation, "vqa_dataset_id") and observation.vqa_dataset_id is not None:
                        vqa_dataset_ids = jnp.asarray(observation.vqa_dataset_id, dtype=jnp.int32)
                        metrics.update(
                            compute_per_vqa_dataset_metrics(
                                per_sample_loss=lang_loss,
                                vqa_dataset_ids=vqa_dataset_ids,
                                vqa_mask=vqa_mask,
                            )
                        )
                if self.enable_prediction_training:
                    metrics.update(
                        compute_sample_specific_metrics(
                            per_sample_loss=lang_loss,
                            sample_mask=pred_mask,
                            prefix="pred_",
                        )
                    )
                metrics.update(
                    compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        sample_mask=lang_mask,
                        prefix="langact_",
                    )
                )

                # Compute per-sample VQA loss weights if per-dataset weights are specified
                if self.enable_vqa_training and self.vqa_loss_weights_by_id is not None:
                    # Get dataset IDs for VQA samples
                    if hasattr(observation, "vqa_dataset_id") and observation.vqa_dataset_id is not None:
                        vqa_dataset_ids = jnp.asarray(observation.vqa_dataset_id, dtype=jnp.int32)
                        # Create per-sample weight array: use per-dataset weight if specified, else default
                        vqa_weights = jnp.full(batch_size, self.vqa_loss_weight, dtype=jnp.float32)
                        for dataset_id, weight in self.vqa_loss_weights_by_id.items():
                            vqa_weights = jnp.where(vqa_dataset_ids == dataset_id, weight, vqa_weights)
                    else:
                        # Fallback to default weight if dataset IDs not available
                        vqa_weights = jnp.full(batch_size, self.vqa_loss_weight, dtype=jnp.float32)
                else:
                    # Use scalar weight for all VQA samples
                    vqa_weights = jnp.full(batch_size, self.vqa_loss_weight, dtype=jnp.float32)

                lang_per_sample_loss += (
                    vqa_weights * lang_loss * vqa_mask
                    + self.prediction_loss_weight * lang_loss * pred_mask
                    + self.language_loss_weight * lang_loss * lang_mask
                )
            else:
                metrics.update(
                    compute_sample_specific_metrics(
                        per_sample_loss=lang_loss,
                        sample_mask=combined_langact_mask,
                        prefix="langact_",
                    )
                )
                lang_per_sample_loss += self.language_loss_weight * lang_loss

        if self.enable_action_training:
            suffix_out = pre_logits[1]
            action_loss, action_metrics = self._compute_action_loss(suffix_out, suffix_inputs["u_t"])
            action_sample_mask = jnp.ones(batch_size, dtype=bool)
            if vqa_mask is not None:
                action_sample_mask = jnp.logical_and(action_sample_mask, jnp.logical_not(vqa_mask))
            if pred_mask is not None:
                action_sample_mask = jnp.logical_and(action_sample_mask, jnp.logical_not(pred_mask))
            action_sample_mask_f = action_sample_mask.astype(jnp.float32)
            action_per_sample_loss += self.action_loss_weight * action_loss * action_sample_mask_f
            action_metrics["action_loss"] = jnp.sum(action_loss * action_sample_mask_f) / jnp.maximum(
                jnp.sum(action_sample_mask_f), 1.0
            )
            metrics.update(action_metrics)

        # Add main metrics to dict
        total_per_sample_loss = lang_per_sample_loss + action_per_sample_loss
        if effective_verbose_mode:
            metrics["per_sample_loss"] = total_per_sample_loss

        # Compute final loss with correct normalization
        # When samples are masked out, their loss is 0 and shouldn't be counted in denominator
        if self.enable_action_training:
            action_term = jnp.sum(action_per_sample_loss) / jnp.maximum(jnp.sum(action_sample_mask_f), 1.0)
            if self.enable_langact_training:
                if observation.sample_mask is not None:
                    num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
                    lang_term = jnp.sum(lang_per_sample_loss) / num_active_samples
                else:
                    lang_term = jnp.mean(lang_per_sample_loss)
            else:
                lang_term = 0.0
            final_loss = lang_term + action_term
        elif self.enable_langact_training and observation.sample_mask is not None:
            # Only langact training with sample masking: divide by number of active samples
            num_active_samples = jnp.maximum(jnp.sum(observation.sample_mask), 1.0)
            final_loss = jnp.sum(total_per_sample_loss) / num_active_samples
        else:
            # No masking or fallback: use mean over all samples
            final_loss = jnp.mean(total_per_sample_loss)

        # Add augmented images to metrics if requested
        if augmented_images is not None:
            metrics["augmented_images"] = augmented_images

        return final_loss, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: CoTObservation | Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation(
            None, observation, train=False, image_keys=self.image_keys, aug_wrist_image=self.aug_wrist_image
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            gemma_out, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = gemma_out[1]
            assert gemma_out[0] is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_tokens(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 390,
        temperature: float = 0.0,
    ) -> _model.Actions:
        observation = preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
            aug_wrist_image=self.aug_wrist_image,
        )

        # Embed only the VLM (expert 0) prefix tokens (images + text).
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)

        # Right-align sequences so padded tokens live on the left and we can use prefix_start logic.
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = _pi0_fast.left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # First fill KV cache with the prefix; pad attention to the full cache size for flexible expert-0 decoding.
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        pre_logits, kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None] if self.enable_action_training else [prefix_token_embeddings],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None, None] if self.enable_action_training else [None],
        )

        # prepare decoding -- final logit decodes the first token
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

            # Track EOS per batch element; stop when all sequences have seen EOS.
            eos_token = jnp.squeeze(token, axis=-1)
            eos_mask = eos_mask | (eos_token == self.EOS_TOKEN)
            all_eos = jnp.all(eos_mask)

            # Decode one step using only expert 0.
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
            )
            last_logit = self.PaliGemma.llm(last_prelogit[0], method="decode")

            return rng, last_logit, output_tokens, kv_cache, eos_mask, step + 1

        def cond(carry):
            _, _, _, _, eos_mask, step = carry
            return (~jnp.all(eos_mask)) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, jnp.zeros((last_logit.shape[0],), dtype=bool), 0)
        )
        return output_tokens
