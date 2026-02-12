# LAP: Language-Action Pre-Training Enables Zero-Shot Cross-Embodiment Transfer

### [<a href="https://lap-vla.github.io/" target="_blank">Website</a>] [<a href="https://arxiv.org/abs/2602.10556" target="_blank">Paper</a>] [<a href="https://huggingface.co/collections/lihzha/lap" target="_blank">Checkpoints</a>]

<p align="center">
  <img src="assets/teaser.gif" width="600">
</p>

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:lihzha/lap.git
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.
After installing uv, set up the environment with:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Real-Robot Evaluation

Example inference script: [scripts/real_robot/droid_main.py](scripts/real_robot/droid_main.py)

Download the LAP checkpoint from [lihzha/LAP-3B](https://huggingface.co/lihzha/LAP-3B) and place it at:

```bash
./checkpoint/lap
```

### 1. Start the policy server

```bash
JAX_PLATFORMS=cuda uv run --group cuda scripts/serve_policy.py policy:checkpoint --env=LAP
```

### 2. Run on DROID

1. Install the latest DROID package on both the control laptop and the NUC.
2. Activate the DROID conda environment on the control laptop.
3. Install the OpenPI client used to connect to the policy server:
   `cd third_party/openpi/packages/openpi-client && pip install -e .`
4. Install `tyro` for CLI parsing:
   `pip install tyro`

```bash
python scripts/real_robot/droid_main.py \
  --external_camera=right \
  --left_camera_id=<left_camera_id> \
  --right_camera_id=<right_camera_id> \
  --wrist_camera_id=<wrist_camera_id>
```

To add support for another robot, use [scripts/real_robot/franka_main.py](scripts/real_robot/franka_main.py) as a reference.

## LIBERO Evaluation

Download the LIBERO checkpoint from [lihzha/LAP-3B-Libero](https://huggingface.co/lihzha/LAP-3B-Libero) and place it at:

```bash
./checkpoint/lap_libero
```

Then follow [scripts/libero/README.md](scripts/libero/README.md).

## Training

Training is supported on both GPUs and TPUs.

Train on LIBERO with GPUs:

```bash
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<your_data_dir>
```

Train on LIBERO with TPUs:

```bash
uv run scripts/train.py lap_libero --exp-name=lap_libero --data.rlds_data_dir=<your_data_dir>
```

Expected dataset layout:

```text
<your_data_dir>/
  libero_10_no_noops/
  libero_goal_no_noops/
  libero_object_no_noops/
  libero_spatial_no_noops/
```

LIBERO RLDS source dataset: [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)

For custom datasets:
1. Arrange datasets in the same directory structure pattern.
2. Define your data mixture in [src/lap/datasets/utils/mixtures.py](src/lap/datasets/utils/mixtures.py).
3. Train with:

```bash
JAX_PLATFORMS=cuda uv run --group cuda scripts/train.py lap --exp-name=lap_custom --data.rlds_data_dir=<your_data_dir> --data.data-mix=<your_datamix_name>
```

## Acknowledgment

This repository is built on [OpenPI](https://github.com/Physical-Intelligence/openpi).

## Citation

If this codebase helps your research, please cite:

```console
@misc{zha2026laplanguageactionpretrainingenables,
      title={LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer}, 
      author={Lihan Zha and Asher J. Hancock and Mingtong Zhang and Tenny Yin and Yixuan Huang and Dhruv Shah and Allen Z. Ren and Anirudha Majumdar},
      year={2026},
      eprint={2602.10556},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.10556}, 
}
```
