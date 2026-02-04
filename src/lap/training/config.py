"""See _CONFIGS for the list of available configs."""

import abc
import dataclasses
import difflib
import pathlib
from typing import Literal, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.training.config as upstream_config
import openpi.training.optimizer as _optimizer
import openpi.transforms as upstream_transforms
from typing_extensions import override
import tyro

from lap.datasets.utils.helpers import ActionEncoding
from lap.datasets.utils.helpers import NormalizationType
from lap.datasets.utils.helpers import StateEncoding
import lap.models.lap_config as lap_config
import lap.models.model_adapter as _model_adapter
from lap.models.tokenizer import FASTTokenizer
from lap.models.tokenizer import Gemma3FASTTokenizer
from lap.models.tokenizer import Gemma3Tokenizer
from lap.models.tokenizer import PaligemmaTokenizer
import lap.policies.lap_policy as lap_policy
import lap.training.weight_loaders as weight_loaders
from lap.transforms import DetokenizeReasoning
from lap.transforms import ExtractFASTActions
from lap.transforms import TokenizeFASTInputs
from lap.transforms import TokenizePromptAndReasoning

ModelType: TypeAlias = _model_adapter.ExtendedModelType
UpstreamModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


def _to_path(base: str | pathlib.Path, *extra: str) -> pathlib.Path | epath.Path:
    """
    Join `base` with any `extra` segments, returning:
      • `pathlib.Path` for normal file-system paths
      • `epath.Path`   for `gs://` URIs
    """
    base = str(base)  # in case the attr is already a Path object
    if base.startswith("gs://"):
        # epath.Path already mimics pathlib semantics (`/`, `.joinpath`, etc.)
        return epath.Path(base).joinpath(*extra)  # no `.resolve()` on GCS
    return (pathlib.Path(base).joinpath(*extra)).resolve()


def build_lap_model(
    *,
    action_horizon: int = 32,
    max_token_len: int = 110,
    pi05: bool = True,
    discrete_state_input: bool = True,
) -> _model.BaseModelConfig:
    """Convenience helper for common LAP model instantiations."""
    return lap_config.LAPConfig(
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        discrete_state_input=discrete_state_input,
    )


def build_cosine_lr(
    *,
    warmup_steps: int = 5_000,
    peak_lr: float = 1e-4,
    decay_steps: int = 40_000,
    decay_lr: float = 1e-4,
) -> _optimizer.LRScheduleConfig:
    """Shared cosine LR schedule used by most experiments."""
    return _optimizer.CosineDecaySchedule(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
    )


@dataclasses.dataclass(frozen=True)
class DataConfig(upstream_config.DataConfig):
    shuffle_buffer_size: int = 1_000_000
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    # Validation controls for RLDS-CoT dataset splitting/visualization
    val_max_samples: int | None = None
    val_fraction: float | None = 0.025
    use_wrist_image: bool = True
    wrist_image_dropout_prob: float = 0.1
    state_encoding: StateEncoding = StateEncoding.POS_EULER
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    # Normalization type for actions and proprioceptive state.
    # CLI: --data.action_proprio_normalization_type {normal|bounds|bounds_q99}
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99
    resize_resolution: tuple[int, int] = (224, 224)
    force_recompute_stats: bool = False
    want_full_determinism: bool = False
    data_mix: str | None = "oxe_magic_soup"
    balance_weights: bool = True
    rlds_data_dir: str = "./data"

    # Augmentation parameters
    aggressive_aug: bool = False
    aug_wrist_image: bool = True
    random_base_prob: float = 0.5
    random_mask_prob: float = 0.2
    not_rotate_wrist_prob: float = 0.0
    use_rough_scale: bool = False

    # Language action format
    language_action_format_name: str = "verbose_eef_with_rotation"
    # Transform behavior mode for sample/output transforms.
    # Use "vla0" only for VLA-0-format transform behavior.
    transform_strategy: Literal["standard", "vla0"] = "standard"
    horizon_seconds: list[float] = dataclasses.field(default_factory=lambda: [1.0])

    # Prediction training parameters
    max_prediction_horizon: int = 30
    pred_prob: float = 0.3  # Probability of converting a frame to prediction sample (after flattening)
    primary_pred_prob: float = 0.8  # Probability of using primary camera (vs wrist) for prediction training

    # Diverse question type configuration for prediction training
    enable_diverse_questions: bool = True  # Enable diverse question types for prediction samples
    # Question type weights (will be normalized). Set to None to use defaults.
    # Available types: delta_motion, task_prediction, direction_classification,
    #                  gripper_prediction, magnitude_estimation, temporal_ordering
    question_type_weights: dict[str, float] | None = None
    # Answer format weights for delta_motion questions. Set to None to use defaults.
    # Available formats: verbose, verbose_with_rotation, compact, compact_with_rotation,
    #                    qualitative, component, json, sentence, direction_only
    delta_motion_format_weights: dict[str, float] | None = None
    # Whether to use diverse prompt sentence variations
    use_diverse_prompts: bool = True

    # VQA bbox dataset parameters
    direction_prob: float = 0.0  # Probability of using direction caption instead of bbox for bbox VQA datasets

    # DROID fields
    droid_dataset_name: Literal["droid", "droid_100"] = "droid"

    # Gemma3-specific fields.
    # Gemma3 tokenizer model is not publicly hosted; users must download it and set this path.
    gemma3_tokenizer_path: str | None = None


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.ModelTransformFactory):
    """Creates model transforms for standard pi0 models."""

    prompt_format: str = "lap"
    prediction_format: str = "default"
    include_outputs: bool = True  # Toggle output transforms (e.g., detokenization)
    fast_tokenizer_path: str = "physical-intelligence/fast"  # KarlP/fast_droid_specialist
    # Gemma3 tokenizer model is not publicly hosted; users must download it and set this path.
    gemma3_tokenizer_path: str | None = None

    def _create_tokenizer(self, model_config: lap_config.LAPConfig, reasoning_mask_prob: float):
        """Create the appropriate tokenizer based on model variant."""
        if "gemma3" in model_config.paligemma_variant:
            if not self.gemma3_tokenizer_path:
                raise ValueError(
                    "Gemma3 model selected, but `gemma3_tokenizer_path` is not set in config. "
                    "The Gemma3 tokenizer is not publicly hosted and must be downloaded manually."
                )
            kwargs = {
                "max_len": model_config.max_token_len,
                "prompt_format": self.prompt_format,
                "prediction_format": self.prediction_format,
                "reasoning_mask_prob": reasoning_mask_prob,
                "tokenizer_model_path": self.gemma3_tokenizer_path,
            }
            # Check if FAST tokenizer is needed
            if getattr(model_config, "use_fast", False):
                kwargs["fast_tokenizer_path"] = self.fast_tokenizer_path
                return Gemma3FASTTokenizer(**kwargs)
            return Gemma3Tokenizer(**kwargs)
        return PaligemmaTokenizer(
            model_config.max_token_len,
            prompt_format=self.prompt_format,
            prediction_format=self.prediction_format,
            reasoning_mask_prob=reasoning_mask_prob,
        )

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.LAP:
            assert isinstance(model_config, lap_config.LAPConfig)
            outputs = []
            if self.include_outputs:
                outputs = [DetokenizeReasoning(self._create_tokenizer(model_config, reasoning_mask_prob=0))]
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizePromptAndReasoning(
                        self._create_tokenizer(model_config, reasoning_mask_prob=model_config.reasoning_mask_prob),
                        discrete_state_input=model_config.discrete_state_input,
                        verbose_mode=model_config.verbose_mode,
                        state_dropout=model_config.state_dropout,
                    ),
                    upstream_transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=outputs,
            )
        if model_config.model_type in (ModelType.LAP_FAST, UpstreamModelType.PI0_FAST):
            assert isinstance(model_config, lap_config.LAPConfig)
            # Create appropriate FAST tokenizer based on model variant
            if "gemma3" in model_config.paligemma_variant:
                if not self.gemma3_tokenizer_path:
                    raise ValueError(
                        "Gemma3 model selected, but `gemma3_tokenizer_path` is not set in config. "
                        "The Gemma3 tokenizer is not publicly hosted and must be downloaded manually."
                    )
                # Use Gemma3FASTTokenizer for Gemma3 models
                tokenizer_kwargs = {
                    "fast_tokenizer_path": self.fast_tokenizer_path,
                    "max_len": model_config.max_token_len,
                    "prompt_format": self.prompt_format,
                    "prediction_format": self.prediction_format,
                    "tokenizer_model_path": self.gemma3_tokenizer_path,
                }
                fast_tokenizer = Gemma3FASTTokenizer(**tokenizer_kwargs)
            else:
                # Use regular FASTTokenizer for PaliGemma models
                fast_tokenizer = FASTTokenizer(
                    fast_tokenizer_path=self.fast_tokenizer_path,
                    max_len=model_config.max_token_len,
                    prompt_format=self.prompt_format,
                    prediction_format=self.prediction_format,
                )
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizeFASTInputs(
                        fast_tokenizer,
                        discrete_state_input=model_config.discrete_state_input,
                        state_dropout=model_config.state_dropout,
                    ),
                    # PadStates(model_config.action_dim),
                ],
                outputs=[
                    ExtractFASTActions(
                        fast_tokenizer,
                        action_horizon=model_config.action_horizon,
                        action_dim=model_config.action_dim,
                    )
                ],
            )

        return super().__call__(model_config)


@dataclasses.dataclass(frozen=True)
class BaseDataConfigFactory(DataConfig, upstream_config.DataConfigFactory, abc.ABC):
    """Base class for all CoT data config factories.

    Provides common implementations for:
    - create_base_config: Extract CoT fields and set up base configuration

    Subclasses must implement:
    - _create_data_transforms: Policy-specific data transformations
    - _create_model_transforms: Model-specific transformations
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create base CoT config with common fields."""
        cot_fields = DataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=None,  # Note: Normalization is handled on dataset level
        )
        return DataConfig(**data)

    @abc.abstractmethod
    def _create_data_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create policy-specific data transforms. Must be implemented by subclasses."""

    @abc.abstractmethod
    def _create_model_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create model-specific transforms. Must be implemented by subclasses."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Template method that orchestrates config creation."""
        base_cfg = self.create_base_config(assets_dirs, model_config)
        data_transforms = self._create_data_transforms(base_cfg, model_config)
        model_transforms = self._create_model_transforms(base_cfg, model_config)

        return dataclasses.replace(
            base_cfg,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDataConfig(BaseDataConfigFactory):
    """
    Config for training on OXE/DROID, using RLDS data format (for efficient training on larger datasets).

    Default values are set for OXE training. Override as needed for other datasets.
    """

    # Override upstream defaults for OXE
    repo_id: str = "oxe"
    asset_id: str = "oxe"

    @override
    def _create_data_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        # Build question config if diverse questions are enabled
        question_config = None
        if base_cfg.enable_diverse_questions:
            from lap.policies.question_types import QuestionConfig

            question_config = QuestionConfig(
                type_weights=base_cfg.question_type_weights,
                delta_motion_format_weights=base_cfg.delta_motion_format_weights,
                use_diverse_prompts=base_cfg.use_diverse_prompts,
            )

        return upstream_transforms.Group(
            inputs=[
                lap_policy.CoTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    wrist_image_dropout_prob=base_cfg.wrist_image_dropout_prob,
                    action_encoding=base_cfg.action_encoding,
                    language_action_format=base_cfg.language_action_format_name,
                    random_mask_prob=base_cfg.random_mask_prob,
                    random_base_prob=base_cfg.random_base_prob,
                    use_rough_scale=base_cfg.use_rough_scale,
                    transform_strategy=base_cfg.transform_strategy,
                    enable_langact_training=model_config.enable_langact_training,
                    enable_diverse_questions=base_cfg.enable_diverse_questions,
                    question_config=question_config,
                )
            ],
            outputs=[
                lap_policy.CoTOutputs(
                    language_action_format=base_cfg.language_action_format_name,
                    transform_strategy=base_cfg.transform_strategy,
                )
            ],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: DataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
            gemma3_tokenizer_path=base_cfg.gemma3_tokenizer_path,
        )(model_config)


@dataclasses.dataclass(frozen=True)
class EmaStage:
    """Defines EMA decay for a specific step range."""

    start_step: int
    end_step: int | None = None  # None means until training ends
    decay: float | None = None  # None disables EMA updates in this range

    def validate(self):
        """Validate stage configuration."""
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError(f"end_step ({self.end_step}) must be > start_step ({self.start_step})")
        if self.decay is not None and not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0.0, 1.0), got {self.decay}")


@dataclasses.dataclass(frozen=True)
class EmaSchedule:
    """Manages EMA decay across multiple step ranges."""

    stages: tuple[EmaStage, ...]

    def __post_init__(self):
        if not self.stages:
            raise ValueError("EmaSchedule must have at least one stage")

        for stage in self.stages:
            stage.validate()

        for i in range(len(self.stages) - 1):
            current_stage = self.stages[i]
            next_stage = self.stages[i + 1]

            if current_stage.end_step is None:
                raise ValueError(
                    f"Stage {i} (starting at {current_stage.start_step}) has end_step=None but is not the last stage"
                )

            if next_stage.start_step < current_stage.end_step:
                raise ValueError(
                    f"Stage {i + 1} (starting at {next_stage.start_step}) overlaps with "
                    f"stage {i} (ending at {current_stage.end_step})"
                )

    def get_stage_for_step(self, step: int) -> EmaStage:
        for stage in self.stages:
            if stage.start_step <= step:
                if stage.end_step is None or step < stage.end_step:
                    return stage
        raise ValueError(
            f"No EMA stage covers step {step}. Available stages: {[(s.start_step, s.end_step) for s in self.stages]}"
        )

    def get_decay_for_step(self, step):
        """JAX-compatible method to get EMA decay and enable flag for a given step."""
        import jax.numpy as jnp

        decay = jnp.asarray(0.0, dtype=jnp.float32)
        enabled = jnp.asarray(False)

        for stage in self.stages:
            in_range = step >= stage.start_step
            if stage.end_step is not None:
                in_range = in_range & (step < stage.end_step)
            else:
                in_range = in_range & (step >= stage.start_step)

            stage_decay = 0.0 if stage.decay is None else stage.decay
            stage_enabled = stage.decay is not None
            decay = jnp.where(in_range, stage_decay, decay)
            enabled = jnp.where(in_range, stage_enabled, enabled)

        return decay, enabled

    def has_ema(self) -> bool:
        return any(stage.decay is not None for stage in self.stages)

    def default_decay(self) -> float | None:
        for stage in self.stages:
            if stage.decay is not None:
                return stage.decay
        return None


@dataclasses.dataclass(frozen=True)
class EmaScheduleChoice:
    """Choice of pre-specified EMA schedules.

    Available schedules:
    - disabled: EMA off
    - constant: EMA on from step 0 with fixed decay
    - delayed: EMA off until start_step, then fixed decay
    - cosine_delayed: EMA off until start_step, then cosine ramp to max decay

    Example:
        # From command line:
        python -m lap.training.train --config lap_droid_lap_v4 \\
            --ema_schedule_choice.kind delayed \\
            --ema_schedule_choice.start_step 10000 \\
            --ema_schedule_choice.decay 0.999
    """

    kind: Literal["disabled", "constant", "delayed", "cosine_delayed"] = "delayed"

    start_step: int = 10000

    def build(self, *, decay: float | None) -> EmaSchedule | None:
        if self.kind == "disabled":
            return None

        if self.kind == "constant":
            if decay is None:
                return None
            return EmaSchedule(stages=(EmaStage(start_step=0, end_step=None, decay=decay),))

        if self.kind == "delayed":
            if decay is None:
                return None
            if self.start_step <= 0:
                return EmaSchedule(stages=(EmaStage(start_step=0, end_step=None, decay=decay),))
            return EmaSchedule(
                stages=(
                    EmaStage(start_step=0, end_step=self.start_step, decay=None),
                    EmaStage(start_step=self.start_step, end_step=None, decay=decay),
                )
            )

        if self.kind == "cosine_delayed":
            return None

        raise ValueError(f"Unsupported EMA schedule kind: {self.kind}")


@dataclasses.dataclass(frozen=True)
class TrainConfig(upstream_config.TrainConfig):
    # Override upstream defaults
    project_name: str = "lap"
    weight_loader: weight_loaders.WeightLoaderChoice = dataclasses.field(
        default_factory=weight_loaders.WeightLoaderChoice
    )
    model: _model.BaseModelConfig = dataclasses.field(default_factory=build_lap_model)
    data: DataConfig = dataclasses.field(default_factory=RLDSDataConfig)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=build_cosine_lr)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=0.0001)
    )
    num_train_steps: int = 40_000
    save_interval: int = 1000
    log_interval: int = 50
    keep_period: int | None = 5000
    resume: bool = True
    seed: int = 0
    ema_decay: float | None = 0.999
    ema_schedule_choice: EmaScheduleChoice = dataclasses.field(
        default_factory=lambda: EmaScheduleChoice(kind="cosine_delayed", start_step=5000)
    )
    checkpoint_base_dir: str = "./checkpoints"
    # New field
    use_validation: bool = False
    val_interval: int = 2000
    allow_partial_weights: bool = True
    # Evaluation fields
    eval_checkpoint_step: int | None = None
    eval_checkpoint_steps: list[int] | None = None  # List of specific checkpoint steps to evaluate
    eval_all_checkpoints: bool = True  # If True, evaluate all available checkpoints sequentially
    eval_start_from_step: int | None = None  # If set, skip checkpoints before this step (useful for resuming)
    num_eval_batches: int | None = 500
    eval_use_ema: bool = True
    eval_split: Literal["val", "train"] = "val"

    @property
    def ema_schedule(self) -> EmaSchedule | None:
        """Build EMA schedule from the choice configuration."""
        return self.ema_schedule_choice.build(decay=self.ema_decay)

    def get_ema_init(self) -> tuple[float | None, bool]:
        """Return the initial EMA decay and whether EMA params should be initialized."""
        if self.ema_schedule_choice.kind == "cosine_delayed":
            if self.ema_decay is None:
                return None, False
            return 0.0, True
        schedule = self.ema_schedule
        if schedule is None:
            return self.ema_decay, self.ema_decay is not None
        stage0 = schedule.get_stage_for_step(0)
        return stage0.decay, schedule.has_ema()

    def get_ema_decay_for_step(self, step):
        """Return EMA decay and enabled flag for a given step (JAX-compatible)."""
        if self.ema_schedule_choice.kind == "cosine_delayed":
            import jax.numpy as jnp

            max_decay = self.ema_decay
            if max_decay is None:
                return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(False)
            start_step = self.ema_schedule_choice.start_step
            max_step = self.num_train_steps
            duration = jnp.maximum(max_step - start_step, 1)
            progress = (step - start_step) / duration
            progress = jnp.clip(progress, 0.0, 1.0)
            decay = max_decay * (1.0 - jnp.cos(jnp.pi * progress)) / 2.0
            enabled = step >= start_step
            return decay, enabled

        schedule = self.ema_schedule
        if schedule is not None:
            return schedule.get_decay_for_step(step)

        if self.ema_decay is None:
            import jax.numpy as jnp

            return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(False)

        import jax.numpy as jnp

        return jnp.asarray(self.ema_decay, dtype=jnp.float32), jnp.asarray(True)

    @property
    @override
    def assets_dirs(self) -> pathlib.Path | epath.Path:
        """Assets directory (works for local paths and gs://…)."""
        return _to_path(self.assets_base_dir, self.name)

    @property
    @override
    def checkpoint_dir(self) -> pathlib.Path | epath.Path:
        """Checkpoint directory (local or Cloud Storage)."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return _to_path(self.checkpoint_base_dir, self.name, self.exp_name)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="lap",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=180,
            enable_action_training=True,
            stop_action_to_vlm_grad=False,
        ),
        batch_size=2048,
    ),
    TrainConfig(
        name="pi05_replicated",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=220,
            use_fast=True,
            enable_action_training=True,
            stop_action_to_vlm_grad=True,
        ),
        batch_size=2048,
    ),
    TrainConfig(
        name="pi0_replicated",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=220,
            enable_action_training=True,
            enable_langact_training=False,
        ),
        batch_size=2048,
    ),
    TrainConfig(
        name="lap_gemma3_4b",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            enable_action_training=True,
            enable_langact_training=True,
            max_token_len=800,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
        ),
        batch_size=2048,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3"),
    ),
    TrainConfig(
        name="fast_gemma3_4b",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            enable_action_training=False,
            enable_langact_training=True,
            max_token_len=800,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
            use_fast=True,
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3"),
        batch_size=2048,
    ),
    TrainConfig(
        name="lap_gemma3_12b",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            enable_action_training=True,
            enable_langact_training=True,
            max_token_len=800,
            paligemma_variant="gemma3_12b",
            action_expert_variant="gemma3_300m_48",
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="checkpoints/gemma3-12b-it"),
        batch_size=2048,
    ),
    TrainConfig(
        name="lap_gemma3_27b",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            enable_action_training=True,
            enable_langact_training=True,
            max_token_len=800,
            paligemma_variant="gemma3_27b",
            action_expert_variant="gemma3_300m_62",
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="checkpoints/gemma3-27b-it"),
        batch_size=2048,
    ),
    # Reference: "VLA-0: Building State-of-the-Art VLAs with Zero Modification"
    TrainConfig(
        name="vla0_replicated",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=390,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=False,
            enable_langact_training=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            prompt_format="vla0_chunked",
        ),
        data=RLDSDataConfig(language_action_format_name="vla0_chunked", transform_strategy="vla0"),
        batch_size=2048,
    ),
    TrainConfig(
        name="vla0_replicated_libero",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=390,
            enable_action_training=False,
            enable_langact_training=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            prompt_format="vla0_chunked",
            reasoning_mask_prob=0.2,
        ),
        data=RLDSDataConfig(
            shuffle_buffer_size=100000,
            repo_id="libero",
            asset_id="libero",
            data_mix="libero_finetune",
            val_fraction=0.0,
            language_action_format_name="vla0_chunked",
            transform_strategy="vla0",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=40_000,
            decay_lr=5e-5,
        ),
        save_interval=2000,
        keep_period=2000,
        num_train_steps=40_001,
        batch_size=256,
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=1000),
    ),
    TrainConfig(
        name="lap_libero",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            enable_action_training=True,
            stop_action_to_vlm_grad=True,
            language_loss_weight=0.4,
            enable_image_augmentation=False,
        ),
        data=RLDSDataConfig(
            shuffle_buffer_size=100000,
            repo_id="libero",
            asset_id="libero",
            data_mix="libero_finetune",
            val_fraction=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=40_000,
            decay_lr=5e-5,
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="checkpoints/lap/params",
        ),
        save_interval=2000,
        keep_period=2000,
        num_train_steps=40_001,
        batch_size=256,
        ema_schedule_choice=EmaScheduleChoice(kind="constant"),
    ),
    TrainConfig(
        name="lap_cotrain",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=220,
            enable_action_training=True,
            enable_prediction_training=True,
            stop_action_to_vlm_grad=True,
        ),
        batch_size=2048,
    ),
    TrainConfig(
        name="lap_gemma3_4b_libero",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            enable_action_training=True,
            enable_langact_training=True,
            max_token_len=800,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
            language_loss_weight=0.4,
            enable_image_augmentation=False,
        ),
        batch_size=2048,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="checkpoints/gemma3-4b-it"),
        data=RLDSDataConfig(
            shuffle_buffer_size=100000,
            repo_id="libero",
            asset_id="libero",
            data_mix="libero_finetune",
            val_fraction=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=40_000,
            decay_lr=5e-5,
        ),
        save_interval=2000,
        keep_period=2000,
        num_train_steps=40_001,
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=1000),
    ),
    *upstream_config._CONFIGS,  # noqa: SLF001
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name.

    Args:
        config_name: Name of the config to retrieve

    Returns:
        The requested TrainConfig

    Examples:
        get_config("lap")
    """
    if config_name in _CONFIGS_DICT:
        return _CONFIGS_DICT[config_name]

    # Config not found - provide helpful error message
    closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=3, cutoff=0.0)
    closest_str = f" Did you mean one of: {', '.join(repr(c) for c in closest)}?" if closest else ""

    raise ValueError(f"Config '{config_name}' not found.{closest_str}")
