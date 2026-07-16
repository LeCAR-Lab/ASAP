# ASAP Isaac Gym smoke reproduction

## Goal

Verify that the upstream G1 locomotion stack can initialize on GPU without
running a baseline training iteration. This is a pre-migration baseline for a
future R1 robot port.

## Environment

- Repository revision: `df5320c`
- Conda environment: `asap`
- Python: 3.8.20
- PyTorch: 2.4.1+cu121
- Torchvision: 0.19.1+cu121
- Isaac Gym: Preview 4 (`1.0rc4` package metadata)
- GPU used for verification: NVIDIA GeForce RTX 4090
- NVIDIA driver: 580.76.05

Create the environment:

```bash
conda env create -f reproduction/asap-isaacgym-conda.yaml
conda activate asap

pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -e /path/to/isaacgym/python --no-deps
pip install scipy==1.10.1 pyyaml imageio ninja
pip install -e . -e isaac_utils
```

Isaac Gym Preview 4 can be downloaded separately with:

```bash
curl -L -o IsaacGym_Preview_4_Package.tar.gz \
  https://developer.nvidia.com/isaac-gym-preview-4
```

Keep the archive and extracted SDK outside this repository.

## Verification

Dependency consistency:

```bash
python -m pip check
```

Result: `No broken requirements found.`

The Isaac Gym GPU PhysX smoke test created a sphere on a plane, enabled the GPU
pipeline, advanced 120 simulation steps, and read the final state through the
tensor API. Result:

```text
ISAAC_GYM_SMOKE_OK NVIDIA GeForce RTX 4090 steps=120 z=0.102
```

The ASAP G1 stack was initialized without training using:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
  +simulator=isaacgym \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_g1_locomotion \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  num_envs=1 \
  project_name=ASAPSmoke \
  experiment_name=G1InitOnly \
  headless=True \
  use_wandb=False \
  algo.config.num_learning_iterations=0
```

Result: one G1 environment was created with GPU PhysX and the GPU pipeline.
ASAP resolved an 81-dimensional actor observation, an 81-dimensional critic
observation, and 23 actions. PPO and rollout storage initialized successfully.
No learning iteration ran.

The bundled ONNX models also load with ONNX Runtime:

- Decoupled locomotion: input `[1, 500]`, output `[1, 12]`
- CR7 mimic: input `[1, 380]`, output `[1, 23]`

## Local outputs

- Isaac Gym SDK: `/home/HDD/czc/deps/asap/isaacgym`
- Isaac Gym archive: `/home/HDD/czc/deps/asap/IsaacGym_Preview_4_Package.tar.gz`
- Initialization output: `logs/ASAPSmoke/20260716_072756-G1InitOnly-locomotion-g1_29dof_anneal_23dof`

The output directory is ignored by Git. It contains the resolved config, short
logs, TensorBoard metadata, and an untrained 1.7 MB `model_0.pt` emitted by the
upstream zero-iteration code path.

## Limits and R1 migration notes

- Isaac Gym Preview 4 and this Python 3.8 stack do not support RTX 5090
  (`sm_120`). Use an RTX 4090 (`sm_89`) for this baseline.
- This stage verifies simulator and G1 environment initialization, not policy
  quality or baseline training.
- The current robot contract is coupled to the G1 URDF/MJCF assets, 29-DoF
  state, 23 controlled joints, joint/body names, default pose, PD gains, limits,
  observation dimensions, reward contact bodies, and ONNX action dimensions.
- An R1 port should first add an R1 robot/asset config and a deterministic joint
  mapping, then validate reset and zero-action stepping before adapting rewards
  or reusing policy checkpoints.
