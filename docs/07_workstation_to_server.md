# 7. Workstation to GPU server

This stage has one purpose: reproduce the accepted workstation inputs on the
GPU server and prove that the server can train them. It does not choose a model
recipe, evaluate official validation, calibrate probabilities, or return final
artifacts. Those operations belong to steps 8--10.

## Entry and exit gates

Enter only after all six dataset roots validate on the workstation and the
local tiny-overfit plus both bounded smoke runs complete. Exit when:

1. the server has the exact frozen Git commit;
2. all six complete dataset roots have transferred and validate;
3. the pinned SegFormer snapshot is cached;
4. the host-matched Python environment passes a real CUDA kernel check; and
5. the deterministic tiny-overfit acceptance run succeeds on one server GPU.

Habitat-Lab, Habitat-Sim, raw HM3D assets, audits, the pilot, and workstation
run directories do not need to move to the training-only server.

## 7.1 Complete the local training acceptance suite

Run all three from the workstation training environment after this reporting
upgrade. Fresh collision-safe run directories are intentional. Generic form:

```bash
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd /path/to/hm3d-semseg

hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline_smoke.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_moderately_balanced_smoke.yaml \
  --local-config configs/local.yaml
```

Current workstation form:

```bash
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd /home/joaocb2002/projects/hm3d-semseg

hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline_smoke.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer_b2_moderately_balanced_smoke.yaml \
  --local-config configs/local.yaml
```

Use the exact directories printed by training. Open each
`ACTUAL_RUN/report/index.html`; tiny overfit should nearly memorize all four
views, and both smoke runs should finish held-out evaluation with finite losses
and aligned ten-view train/development contact sheets. Compare the smoke paths
as an integration check. Generic form:

```bash
hm3d-semseg compare-runs \
  --run /workstation/runs/ACTUAL_BASELINE_SMOKE \
  --run /workstation/runs/ACTUAL_BALANCED_SMOKE \
  --output /workstation/runs/comparisons/smoke
```

Current workstation form:

```bash
hm3d-semseg compare-runs \
  --run /home/joaocb2002/hm3d-semseg-data/runs/ACTUAL_BASELINE_SMOKE \
  --run /home/joaocb2002/hm3d-semseg-data/runs/ACTUAL_BALANCED_SMOKE \
  --output /home/joaocb2002/hm3d-semseg-data/runs/comparisons/smoke
```

The smoke comparison proves reporting and evaluation mechanics; its 512/256
subsets are too small to select the final recipe.

## 7.2 Freeze the workstation source

Run on the workstation after the local smoke runs:

```bash
cd /path/to/hm3d-semseg
git status --short
git diff --check
python -m pytest -m unit
ruff check .
git rev-parse HEAD
```

Current workstation form:

```bash
cd /home/joaocb2002/projects/hm3d-semseg
git status --short
git diff --check
python -m pytest -m unit
ruff check .
git rev-parse HEAD
```

Review, commit, and push every intended source, experiment-YAML, test, and
documentation change. Do not commit `configs/local.yaml`, datasets, caches,
runs, or weights. Record the resulting commit SHA. Do not migrate a dirty
working tree.

## 7.3 Prepare persistent server storage

Prefer persistent project/work storage or approved scratch. Do not place long
runs under an ephemeral `TMPDIR`. Generic layout:

```text
/server/project/root/
├── repository/
├── data/
├── runs/
├── cache/
└── logs/
```

Current `knuth` layout:

```text
/workspace/
├── repository/
├── data/
├── runs/
├── cache/
└── logs/
```

Before creating environments or copying data, record the server identity,
storage, GPU, scheduler, and available tools:

```bash
hostname -f
whoami
pwd
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v sbatch
command -v qsub
command -v bsub
df -h .
quota -s
printenv | grep -E '^(SCRATCH|WORK|PROJECT|TMPDIR)='
git --version
command -v conda
command -v micromamba
command -v mamba
command -v rsync
command -v tmux
python3 --version
```

Generic directory creation:

```bash
mkdir -p \
  /server/project/root/repository \
  /server/project/root/data \
  /server/project/root/runs \
  /server/project/root/cache \
  /server/project/root/logs
```

Current `knuth` form:

```bash
mkdir -p \
  /workspace/repository \
  /workspace/data \
  /workspace/runs \
  /workspace/cache \
  /workspace/logs

df -h /workspace
```

The six datasets occupy about 17 GB in the current render. `/workspace` had
approximately 98 GB free before transfer, leaving useful headroom for the
environment, model cache, and run artifacts; continue monitoring it with
`df -h /workspace`.

## 7.4 Transfer the six immutable dataset roots

Generic resumable command, run from the workstation:

```bash
rsync -a \
  --partial \
  --info=progress2 \
  --checksum \
  /local/generated/train-v1 \
  /local/generated/development-v1 \
  /local/generated/train-all-v1 \
  /local/generated/official-val-v1 \
  /local/generated/calibration-fit-v1 \
  /local/generated/calibration-evaluation-v1 \
  USER@SERVER:/server/project/root/data/
```

Current workstation-to-`knuth` command:

```bash
rsync -a \
  --partial \
  --info=progress2 \
  --checksum \
  --rsync-path=/workspace/miniconda/envs/hm3d-transfer/bin/rsync \
  /home/joaocb2002/hm3d-semseg-data/generated/train-v1 \
  /home/joaocb2002/hm3d-semseg-data/generated/development-v1 \
  /home/joaocb2002/hm3d-semseg-data/generated/train-all-v1 \
  /home/joaocb2002/hm3d-semseg-data/generated/official-val-v1 \
  /home/joaocb2002/hm3d-semseg-data/generated/calibration-fit-v1 \
  /home/joaocb2002/hm3d-semseg-data/generated/calibration-evaluation-v1 \
  joao_branco@knuth:/workspace/data/
```

The explicit `--rsync-path` is required because `knuth` uses the small
`hm3d-transfer` Conda environment for its server-side `rsync`. Do not add
`--delete`. Rerun the identical command after completion; a near-zero second
transfer verifies content identity. Preserve every dataset's root YAML, JSON,
JSONL, validation artifacts, camera profile, scene directories, RGB files, and
masks.

If the remote side reports `rsync: command not found`, create the isolated
transfer utility first. Generic server form:

```bash
/path/to/miniconda/bin/conda create \
  --name hm3d-transfer \
  --channel conda-forge \
  rsync \
  --yes
```

Current `knuth` form:

```bash
/workspace/miniconda/bin/conda create \
  --name hm3d-transfer \
  --channel conda-forge \
  rsync \
  --yes
```

Verify it from the workstation:

```bash
ssh joao_branco@knuth \
  '/workspace/miniconda/envs/hm3d-transfer/bin/rsync --version'
```

## 7.5 Clone the exact source commit

Generic server commands:

```bash
cd /server/project/root/repository
git clone REPOSITORY_URL hm3d-semseg
cd hm3d-semseg
git checkout --detach FROZEN_COMMIT_SHA
git status --short
git rev-parse HEAD
```

Current server form for the source commit used by the running baseline/balanced
development experiments:

```bash
cd /workspace/repository
git clone https://github.com/joaocb2002/hm3d_semseg.git hm3d-semseg
cd /workspace/repository/hm3d-semseg
git checkout --detach fe8ad759478c90adf568e6cdd8bf63d28f61c0dc
git status --short
git rev-parse HEAD
```

The status must be empty and the SHA must equal the workstation value. If the
server cannot reach the Git remote, transfer a Git bundle containing the frozen
commit instead of copying an uncommitted directory.

## 7.6 Create the host-specific training environment

Run host detection on the actual GPU machine, not a restricted login node. The
trainer is single-process and single-GPU; expose or request exactly one GPU.

Generic commands:

```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda create -n hm3d-semseg-train python=3.10 --yes
conda activate hm3d-semseg-train
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd /server/project/root/repository/hm3d-semseg
python -m pip install -e .
hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply --run-tests
```

Current `knuth` form:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda create -n hm3d-semseg-train python=3.10 --yes
conda activate hm3d-semseg-train
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
cd /workspace/repository/hm3d-semseg
python -m pip install -e .
hm3d-semseg install-training-env
hm3d-semseg install-training-env --apply --run-tests
```

The first installer call is a read-only plan. Inspect it before applying. Keep
the resulting environment unchanged across baseline, balanced, final,
evaluation, and calibration runs.

For a directly managed server without Slurm/PBS/LSF, first confirm that direct
GPU use is permitted, inspect `nvidia-smi`, and mask one free GPU for the whole
session:

```bash
export CUDA_VISIBLE_DEVICES=GPU_INDEX
```

Current example when physical GPU 0 is assigned:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Inside the process that selected physical GPU appears as `cuda:0`.

## 7.7 Create `configs/local.yaml` on the server

Do not copy the workstation file. Generic training-only form:

```yaml
paths:
  generated_data_root: /server/project/root/data
  runs_root: /server/project/root/runs
  cache_root: /server/project/root/cache

camera:
  profile: /server/project/root/data/train-v1/camera_profile.yaml
  allow_mismatch: false

model:
  revision: "PINNED_HUGGING_FACE_COMMIT"
  local_files_only: true

training:
  device: auto
```

Current `knuth` form:

```yaml
paths:
  generated_data_root: /workspace/data
  runs_root: /workspace/runs
  cache_root: /workspace/cache

camera:
  profile: /workspace/data/train-v1/camera_profile.yaml
  allow_mismatch: false

model:
  revision: "de01bae28967510f9ddd496c60a969357195400c"
  local_files_only: true

training:
  device: auto
```

The pinned revision is the workstation-resolved
`nvidia/segformer-b2-finetuned-ade-512-512` commit. Download exactly that
snapshot:

```bash
hm3d-semseg download-model \
  --local-config configs/local.yaml \
  --model-id nvidia/segformer-b2-finetuned-ade-512-512 \
  --revision PINNED_HUGGING_FACE_COMMIT
```

Current command:

```bash
hm3d-semseg download-model \
  --local-config configs/local.yaml \
  --model-id nvidia/segformer-b2-finetuned-ade-512-512 \
  --revision de01bae28967510f9ddd496c60a969357195400c
```

The printed `resolved_revision` must match exactly. `local_files_only: true`
then prevents later training from silently changing the pretrained input.

## 7.8 Validate the transferred roots

The generic form is `hm3d-semseg validate-dataset --dataset DATASET_ROOT`.
For the current server, run once for every root:

```bash
hm3d-semseg validate-dataset --dataset /workspace/data/train-v1
hm3d-semseg validate-dataset --dataset /workspace/data/development-v1
hm3d-semseg validate-dataset --dataset /workspace/data/train-all-v1
hm3d-semseg validate-dataset --dataset /workspace/data/official-val-v1
hm3d-semseg validate-dataset --dataset /workspace/data/calibration-fit-v1
hm3d-semseg validate-dataset --dataset /workspace/data/calibration-evaluation-v1
```

Validation checks root contracts and fully decodes every RGB/mask pair. This is
why it is slower than a checksum but catches validly transferred yet corrupt or
semantically invalid files.

## 7.9 Repeat tiny overfit as server acceptance

Use a durable terminal because SSH disconnection must not kill the run:

```bash
tmux new -s hm3d-acceptance
```

Inside the current server session:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
cd /workspace/repository/hm3d-semseg

hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml
```

Detach with `Ctrl-b`, then `d`; reattach with `tmux attach -t
hm3d-acceptance`. Accept the server only if loss falls, the selected subset is
nearly memorized, qualitative masks align, CUDA memory/throughput are sensible,
and both `checkpoints/best` and `checkpoints/last` exist under `/workspace/runs`.
This test says nothing about generalization.

For a run produced by the current code, open its generated report directly in
the VS Code file explorer:

```text
/workspace/runs/ACTUAL_OVERFIT_RUN/report/index.html
```

Current first-run path:

```text
/workspace/runs/overfit_tiny/report/index.html
```

Tiny overfit shows all four selected training views. It has no development
dataset, so an empty development section is intentional rather than a missing
evaluation.

Next: [develop and refit the model on the server](08_server_training.md).
