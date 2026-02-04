from __future__ import annotations

import dataclasses
from typing import Any

from openpi.models import model as _model
import openpi.policies.policy as _policy
from openpi.training import config as _config
import openpi.transforms as up_transforms

from lap.policies.policy_adapter import ARPolicy
from lap.policies.transforms.output_transforms import CoTOutputs
import lap.transforms as transforms


@dataclasses.dataclass(frozen=True)
class OutputTransformAssembler:
    """Build output transform stack based on configured transform strategy."""

    transform_strategy: str = "standard"

    def build(
        self,
        *,
        data_config,
        repack_transforms: up_transforms.Group,
        norm_stats: dict[str, up_transforms.NormStats],
        normalization_type,
    ) -> list:
        if self.transform_strategy == "vla0":
            return self._build_vla0_outputs(
                data_config=data_config,
                repack_transforms=repack_transforms,
                norm_stats=norm_stats,
                normalization_type=normalization_type,
            )
        return self._build_standard_outputs(
            data_config=data_config,
            repack_transforms=repack_transforms,
            norm_stats=norm_stats,
            normalization_type=normalization_type,
        )

    def _build_vla0_outputs(
        self,
        *,
        data_config,
        repack_transforms: up_transforms.Group,
        norm_stats: dict[str, up_transforms.NormStats],
        normalization_type,
    ) -> list:
        output_transforms_list = list(data_config.model_transforms.outputs)
        for transform in data_config.data_transforms.outputs:
            if isinstance(transform, CoTOutputs):
                output_transforms_list.append(
                    dataclasses.replace(
                        transform,
                        norm_stats=norm_stats,
                        normalization_type=str(normalization_type.value)
                        if hasattr(normalization_type, "value")
                        else str(normalization_type),
                    )
                )
            else:
                output_transforms_list.append(transform)
        output_transforms_list.extend(repack_transforms.outputs)
        return output_transforms_list

    def _build_standard_outputs(
        self,
        *,
        data_config,
        repack_transforms: up_transforms.Group,
        norm_stats: dict[str, up_transforms.NormStats],
        normalization_type,
    ) -> list:
        return [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, normalization_type=normalization_type),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ]


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    *,
    repack_transforms: up_transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, up_transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    # Lazy import to avoid loading TensorFlow at module import time
    from lap.shared.download import maybe_download

    repack_transforms = repack_transforms or up_transforms.Group()
    checkpoint_dir = maybe_download(str(checkpoint_dir))
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params"))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # Lazy import to avoid loading TensorFlow and other heavy dependencies at module import time
        from lap.training import checkpoints as _checkpoints

        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets")

    normalization_type = getattr(data_config, "action_proprio_normalization_type", "normal")
    transform_strategy = getattr(data_config, "transform_strategy", "standard")
    output_transforms_list = OutputTransformAssembler(transform_strategy=transform_strategy).build(
        data_config=data_config,
        repack_transforms=repack_transforms,
        norm_stats=norm_stats,
        normalization_type=normalization_type,
    )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            up_transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, normalization_type=normalization_type),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=output_transforms_list,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=False,
        pytorch_device=None,
    )


def create_trained_policy_ar(*args, sample_kwargs: dict | None = None, **kwargs) -> ARPolicy:
    """Build the standard policy via upstream, then wrap with ARPolicy."""
    base = create_trained_policy(*args, **kwargs)
    return ARPolicy(base, sample_kwargs=sample_kwargs)
