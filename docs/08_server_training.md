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

`configs/experiments/segformer-b2-server/ade20k_recipe.yaml` adapts the official
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
    Path("configs/experiments/segformer-b2-server/ade20k_recipe.yaml"),
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

## 8.5 Run the bounded generalization probe

Before another 160,000-step recipe-development run, execute the checked probe
that tests gentler encoder adaptation on the complete recipe-development data:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b2-server/generalization_probe.yaml \
  --local-config configs/local.yaml
```

On `knuth`, run it from `/workspace/repository/hm3d-semseg`. It trains on all
51,215 `train-v1` frames and evaluates all 3,729 `development-v1` frames. The
2,048-sample setting limits only the final unaugmented training diagnostic; it
does not limit weight updates. The run stops at 48,000 optimizer steps, 15
epochs, or after five epochs without a new development-mIoU best, whichever
comes first.

Inspect these three complete checkpoints when present:

- `checkpoints/best`: highest development known-class mIoU;
- `checkpoints/min_development_loss`: lowest development cross-entropy;
- `checkpoints/last`: final executed epoch.

The train-subset report under `diagnostics/train_subset/` is evaluated with
`checkpoints/best` and provides the missing like-for-like hard-metric evidence
needed to distinguish scene overfitting from a shared train/development ceiling.

The completed `6e-6` probe reached nearly the same held-out mIoU as the
160,000-step reference while keeping development cross-entropy substantially
lower. Its hard metrics nevertheless plateaued and retained a measurable
train/development gap. Preserve that run as the stable comparison; do not
resume it after its learning-rate schedule has reached zero.

### 8.5.1 Run the controlled intermediate-encoder-LR follow-up

The next recipe-development command changes only the encoder learning rate
from `6e-6` to `2e-5`. The head LR, seed, data, augmentations, schedule length,
diagnostics, checkpoint rules, and evaluation set remain identical:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b2-server/generalization_probe_intermediate_lr.yaml \
  --local-config configs/local.yaml
```

On `knuth`, run it from `/workspace/repository/hm3d-semseg`. Expect the run
root `/workspace/runs/segformer_b2_generalization_probe_intermediate_lr`, with
a collision-safe suffix when needed. Compare its `index.html` directly with
the stable probe. Accept the higher LR only if development hard metrics improve
while cross-entropy remains bounded and the deterministic train/development
gap does not materially worsen. Do not inspect `official-val-v1` during this
comparison.

This run is deliberately not accompanied by another subset smoke YAML: the
learning-rate scalar uses the already-tested optimizer path, while a 1,024-view
smoke metric cannot select between learning rates. Configuration unit tests
enforce that LR and run name are the only differences between the two full
probes.

### 8.5.2 Run the controlled CE + Lovasz follow-up

After preserving the two CE probes, run the checked IoU-aligned comparison:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b2-server/generalization_probe_ce_lovasz.yaml \
  --local-config configs/local.yaml
```

On `knuth`, run it from `/workspace/repository/hm3d-semseg`. It restores the
stable probe's `6e-6` encoder LR and keeps its data, seed, augmentations, head
LR, schedule, stopping, checkpoints, and evaluation unchanged. The sole
optimization change is
`0.8 × unweighted cross-entropy + 0.2 × Lovasz-Softmax`. Lovasz is computed at
the native decoder-logit resolution over ground-truth-present known classes
1-40; nearest interpolation preserves the integer target and target 255 is
ignored. The lower resolution avoids making the sort-based loss dominate run
time while CE and all held-out metrics remain full-resolution.

The output is collision-safe under
`/workspace/runs/segformer_b2_generalization_probe_ce_lovasz`. Its raw JSONL,
TensorBoard data, epoch CSV, optimization plot, and overview plot separately
record the total training objective, raw CE component, and raw Lovasz component.
Select `checkpoints/best` by the same development known-class mIoU as every
other development recipe. Judge success against both prior probes: require a
repeatable hard-metric gain without a material regression in development CE,
ObjectNav-six mIoU, or the train/development gap.

### 8.5.3 Run the controlled inverse-square-root follow-up

To isolate class weighting under the stable generalization protocol, run:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b2-server/generalization_probe_inverse_sqrt.yaml \
  --local-config configs/local.yaml
```

On `knuth`, run it from `/workspace/repository/hm3d-semseg`. It is identical to
the stable `6e-6` probe except for inverse-square-root weights computed from the
selected training pixels and the distinct run name. The exact normalized,
capped weight vector and its hash are preserved in the run artifacts. Compare
it with the stable probe, not with a smoke run or the older batch-2 historical
recipes. This closes the class-weighting ablation; it is not a reason to inspect
`official-val-v1` or begin a weight-function sweep.

### 8.5.4 Run the matched SegFormer-B5 probes

The B5 family tests model capacity without changing the training question. Its
four configs correspond exactly to the four B2 generalization probes. Apart
from the model ID, immutable model revision, and run name, each B5 resolved
configuration equals its B2 counterpart. The first comparison deliberately
keeps 512 x 512 crops even though the source B5 checkpoint was ADE-finetuned at
640 x 640; changing architecture and crop resolution together would prevent
causal attribution.

Download the checked B5 snapshot once on `knuth`:

```bash
hm3d-semseg download-model \
  --local-config configs/local.yaml \
  --model-id nvidia/segformer-b5-finetuned-ade-640-640 \
  --revision 739f5d4692954e4a185eac280dec1ba5a7d52f1d
```

Run the stable B5 probe first:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b5-server/generalization_probe.yaml \
  --local-config configs/local.yaml
```

Only if that establishes a useful capacity gain should the three matched
ablations consume additional server time:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b5-server/generalization_probe_intermediate_lr.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer-b5-server/generalization_probe_inverse_sqrt.yaml \
  --local-config configs/local.yaml

hm3d-semseg train \
  --config configs/experiments/segformer-b5-server/generalization_probe_ce_lovasz.yaml \
  --local-config configs/local.yaml
```

B5 has about 84.6 million parameters versus about 27.5 million for B2, so it
will be slower and use substantially more memory. Batch 16 is retained for the
controlled comparison and should fit the 96-GiB assigned GPU. Treat an actual
CUDA out-of-memory error as evidence for a reviewed batch-8/accumulation-2
fallback; do not change batch size during a live run.

## 8.6 Reproduce the historical 160,000-step reference only when needed

The following run is retained for provenance and controlled reproduction. It
is not the next recommended command after the CE+Lovasz experiment:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b2-server/ade20k_recipe.yaml \
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

## 8.7 Monitor and accept development evidence

For the current CE+Lovasz comparison:

```bash
jq '{best: .development.best_known_class_miou,
     final: .development.final,
     optimization: .training.optimization}' \
  /workspace/runs/segformer_b2_generalization_probe_ce_lovasz/records/metrics_summary.json

jq '{epoch, step, primary_metric, best_development_loss}' \
  /workspace/runs/segformer_b2_generalization_probe_ce_lovasz/checkpoints/best/checkpoint.json
```

Use the actual suffixed directory if necessary. Open
`/workspace/runs/ACTUAL_RUN/index.html` through VS Code Remote SSH. Read
the evidence in this order:

1. best development known-class mIoU and whether it improves materially over
   the historical baseline;
2. ObjectNav-six and per-class IoU, especially the prior systematic confusions;
3. global-pixel versus scene-macro mIoU and their per-scene distribution;
4. development cross-entropy and the train/development gap;
5. fixed qualitative development views across epochs;
6. finite gradients, AMP skips, throughput, and peak GPU memory.

Training objective and its CE/Lovasz components diagnose optimization; they do
not select the model.
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

## 8.8 Freeze and run the final refit

Do not create the final experiment until the recipe-development report is
accepted. Record the winning source YAML, its best epoch and step, and its
schedule horizon. On the workstation, create a checked `final.yaml` inside the
winning model family: `configs/experiments/segformer-b2-server/final.yaml` for
B2 or `configs/experiments/segformer-b5-server/final.yaml` for B5. Copy it from
the winning probe—not automatically from the historical 160,000-step file—and
change the dataset and run identity:

```yaml
training:
  datasets:
    train: train-all-v1
    development: null
  max_train_samples: null
  max_development_samples: null
  run_name: SELECTED_MODEL_final
  epochs: FROZEN_FINAL_EPOCHS
  max_optimizer_steps: FROZEN_FINAL_STEPS
  learning_rate_schedule_steps: FROZEN_FINAL_SCHEDULE_STEPS
```

Freeze those values only after recipe selection. Because `train-all-v1` is
larger than `train-v1`, translate the selected duration by full-dataset epochs
so the final refit receives the same intended number of passes; do not blindly
reuse either `48,000` or `160,000`. Record the arithmetic in the final YAML's
comments. Starting from `checkpoints/best` would be wrong: the final run starts
fresh from the same pinned ADE checkpoint with `resume: null` and exposes all
145 training scenes.

Commit and push that final YAML on the workstation, check out its commit on the
server, then run the matching path. For example, if B5 wins:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer-b5-server/final.yaml \
  --local-config configs/local.yaml
```

The expected root follows the checked `run_name`, for example
`/workspace/runs/segformer_b5_final/`. With no development
split, `checkpoints/last` at the frozen step is the protocol checkpoint;
`checkpoints/best` only tracks training loss and is not generalization evidence.

Every run keeps its resolved configuration, source/environment provenance,
append-only metrics, human report, TensorBoard events, diagnostics, and complete
best/last checkpoints below `/workspace/runs`. These artifacts never enter Git.

Next: [evaluate the frozen final checkpoint](09_server_evaluation_and_calibration.md).
