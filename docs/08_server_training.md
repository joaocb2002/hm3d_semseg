# 8. Server training: recipe development and final refit

**Execution location: GPU server.** Start only after step 7 accepts the server.
This stage updates SegFormer weights. It does not use `official-val-v1` and does
not fit a calibration temperature.

There are two separate runs:

1. train on `train-v1`, evaluate every epoch on scene-disjoint
   `development-v1`, and select a checkpoint by development known-class mIoU;
2. start again from the pinned ADE checkpoint and refit the frozen recipe on
   `train-all-v1` for the selected optimizer-step count.

The current `knuth` roots are `/workspace/repository/hm3d-semseg` for source,
`/workspace/data` for datasets, `/workspace/runs` for artifacts, and
`/workspace/cache` for the pinned model snapshot.

## 8.1 Enter a durable one-GPU session

Generic direct-server form:

```bash
tmux new -s hm3d-training
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=GPU_INDEX
cd /server/repository/hm3d-semseg
```

Current `knuth` form when GPU 0 is assigned:

```bash
tmux new -s hm3d-training
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
cd /workspace/repository/hm3d-semseg
```

Detach with `Ctrl-b`, then `d`; list with `tmux ls`; reattach with
`tmux attach -t hm3d-training`. If the site introduces a scheduler, obtain one
GPU through that scheduler instead of launching compute on a login node. The
training command and persistent roots remain the same.

## 8.2 Freeze the loss decision

The completed baseline-versus-inverse-square-root comparison did not show a
meaningful held-out mIoU advantage for class weighting and made several
high-support/ObjectNav-relevant results worse. The next recipe therefore uses
ordinary per-valid-pixel cross-entropy. Lovasz, Dice, focal loss, oversampling,
and taxonomy changes are not added simultaneously with the pipeline correction.
They remain possible future controlled ablations only if the corrected recipe
still has a specific, evidenced failure.

The ADE20K checkpoint does **not** output ADE labels during this project. Loading
replaces its 150-class classifier with a randomly initialized 41-class
classifier; targets come only from the HM3D-to-MPCAT40 mapping. ADE pretraining
supplies transferable features and the rest of the decode head. Confusions such
as stool versus chair or blinds versus window are therefore not old ADE class
IDs leaking into predictions. They reflect visual/taxonomic ambiguity and
cross-scene generalization, which this run addresses first through stronger
scale, crop, and photometric diversity.

## 8.3 Know the recommended recipe

`configs/experiments/segformer_b2_ade20k_recipe.yaml` adapts the official
SegFormer-B2 ADE20K recipe:

| Component | Frozen setting |
|---|---|
| Objective | unweighted cross-entropy; target 255 ignored; class 0 learnable |
| Train geometry | random fit-preserving `(2048, 512)` resize at scale `0.5--2.0`, category-aware `512 x 512` crop, pad with target 255 |
| RGB augmentation | horizontal flip 0.5 and ADE-style photometric distortion |
| Optimizer | AdamW, weight decay 0.01 except normalization/bias parameters |
| Learning rates | pretrained parameters `6e-5`; complete decode head `6e-4` |
| Schedule | 1,500-step linear warm-up from factor `1e-6`, then power-1 polynomial decay to zero at step 160,000 |
| Server batch | 16 on one 96-GiB GPU, AMP enabled, 8 loader workers |
| Selection | highest development known-class mIoU; development never supplies gradients |

The published configuration uses the same resize/crop family, cross-entropy,
AdamW settings, head multiplier, linear warm-up, polynomial decay, and 160k
iteration horizon. This repository intentionally keeps `reduce_labels=false`
because MPCAT40 class 0 is a real `unknown` output, not ADE's background rule.

This is a close protocol adaptation, not a claim of reproducing ADE20K: the
dataset, taxonomy, pretrained starting point, single-GPU execution, and
evaluation partitions differ. Upstream references are the
[official B2 recipe](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py)
and [official data pipeline](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py).

## 8.4 Pull the frozen commit and verify the resolved recipe

Do not pull into an active older training process. After that process finishes,
record its commit and run these commands in the new session.

Generic form:

```bash
cd /server/repository/hm3d-semseg
git status --short
git fetch origin
git checkout COMMITTED_SHA
python -m pytest -m unit
ruff check .
```

Current `knuth` form:

```bash
cd /workspace/repository/hm3d-semseg
git status --short
git fetch origin
git checkout COMMITTED_SHA
python -m pytest -m unit
ruff check .
```

`git status --short` must be empty before checkout. Replace `COMMITTED_SHA` with
the workstation commit produced after this change; do not train an uncommitted
server-only edit.

Inspect the scientific fields without starting training:

```bash
python - <<'PY'
from pathlib import Path
from hm3d_semseg.config import load_config

config = load_config(
    Path("configs/experiments/segformer_b2_ade20k_recipe.yaml"),
    Path("configs/local.yaml"),
)
print(config.training.train_dataset)
print(config.training.development_dataset)
print(config.training.batch_size, config.training.max_optimizer_steps)
print(config.training.learning_rate_schedule, config.training.warmup_steps)
print(config.augmentation)
PY
```

On `knuth`, the first two lines must be `/workspace/data/train-v1` and
`/workspace/data/development-v1`; the batch/step line must be `16 160000`.

## 8.5 Run recipe development

Generic command:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_ade20k_recipe.yaml \
  --local-config configs/local.yaml
```

The current command is identical after entering
`/workspace/repository/hm3d-semseg`. The expected first output is:

```text
/workspace/runs/segformer_b2_ade20k_recipe/
```

If that directory exists, training safely allocates `-002`, `-003`, and so on.
Always record the actual path printed at startup. During the first few hundred
steps, check `nvidia-smi`: memory should be stable, samples/s finite, and GPU
utilization generally high. Do not change batch size during a run. If batch 16
unexpectedly fails on the 96-GiB GPU, stop and commit a reviewed fallback using
batch 8 with gradient accumulation 2; that preserves the effective batch of 16
and the optimizer-step schedule.

The 51,215-sample training manifest gives about 3,201 optimizer steps per full
epoch at batch 16. The 50-epoch guard is slightly longer than needed; the
explicit 160,000-step cap is authoritative and stops the final partial epoch.
Every completed epoch evaluates the complete development manifest and updates
`checkpoints/best` only when known-class mIoU improves.

## 8.6 Monitor and accept development evidence

For the current run:

```bash
jq '{best: .development.best_known_class_miou,
     final: .development.final,
     optimization: .training.optimization}' \
  /workspace/runs/segformer_b2_ade20k_recipe/metrics_summary.json

jq '{epoch, step, primary_metric}' \
  /workspace/runs/segformer_b2_ade20k_recipe/checkpoints/best/checkpoint.json
```

Use the actual suffixed directory if necessary. Open
`/workspace/runs/ACTUAL_RUN/report/index.html` through VS Code Remote SSH. Read
the evidence in this order:

1. best development known-class mIoU and whether it improves materially over
   the historical baseline;
2. ObjectNav-six and per-class IoU, especially the prior systematic confusions;
3. global-pixel versus scene-macro mIoU and their per-scene distribution;
4. development cross-entropy and the train/development gap;
5. fixed qualitative development views across epochs;
6. finite gradients, AMP skips, throughput, and peak GPU memory.

Training cross-entropy diagnoses optimization; it does not select the model.
Calibration metrics, risk-coverage, ECE, and temperature are secondary here and
must not rescue a poor hard segmentation model. A rising development loss with
flat mIoU still indicates worsening probability fit even if the argmax masks
stop changing.

For live inspection, start a second session:

```bash
tmux new -s hm3d-tensorboard
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
tensorboard --logdir /workspace/runs --host 127.0.0.1 --port 6006
```

Forward port 6006 in the VS Code Remote SSH **Ports** panel. TensorBoard is for
live exploration; the JSONL, checkpoint metadata, static report, and resolved
configuration remain the archival evidence.

## 8.7 Freeze and run the final refit

Do not create the final experiment until the recipe-development report is
accepted. Read the exact best optimizer step from
`checkpoints/best/checkpoint.json`. On the workstation, copy
`segformer_b2_ade20k_recipe.yaml` to a new checked
`configs/experiments/segformer_b2_final.yaml` and change only:

```yaml
training:
  datasets:
    train: train-all-v1
    development: null
  max_train_samples: null
  max_development_samples: null
  run_name: segformer_b2_final
  epochs: 50
  max_optimizer_steps: FROZEN_BEST_STEP
  learning_rate_schedule_steps: 160000
```

Keeping `learning_rate_schedule_steps: 160000` is important: it reproduces the
selected learning-rate trajectory while `max_optimizer_steps` stops at the
preselected checkpoint step. Starting from `checkpoints/best` would be wrong;
the final run starts fresh from the same pinned ADE checkpoint with
`resume: null` and exposes all 145 training scenes.

Commit and push that final YAML on the workstation, check out its commit on the
server, then run:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_final.yaml \
  --local-config configs/local.yaml
```

The expected root is `/workspace/runs/segformer_b2_final/`. With no development
split, `checkpoints/last` at the frozen step is the protocol checkpoint;
`checkpoints/best` only tracks training loss and is not generalization evidence.

Every run keeps its resolved configuration, source/environment provenance,
append-only metrics, human report, TensorBoard events, diagnostics, and complete
best/last checkpoints below `/workspace/runs`. These artifacts never enter Git.

Next: [evaluate the frozen final checkpoint](09_server_evaluation_and_calibration.md).
