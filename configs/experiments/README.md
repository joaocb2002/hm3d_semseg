# Experiment configurations

These YAML files configure model-weight training and optional development
evaluation. `hm3d-semseg train` never fits a calibration temperature.

The separate `hm3d-semseg smoke-test` command is a render-to-inference systems
check implemented in Python; it is not a training recipe and does not consume
an experiment YAML. Files ending in `_smoke.yaml` are bounded training
experiments over already-generated datasets.

## Planned sequence

| Config | Train data | Development data | Duration | Loss | Purpose |
|---|---:|---:|---:|---|---|
| `overfit_tiny.yaml` | 4 from `train-v1` | `null` | 50 epochs | cross-entropy | Verify memorization and training mechanics. |
| `segformer_b2_ade20k_recipe_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 2 epochs, at most 1,000 steps | cross-entropy | Exercise the new spatial and optimizer path locally. |
| `segformer_b2_ade20k_recipe.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | at most 160,000 steps | cross-entropy | Historical official-style reference; retain for reproduction, not as the next run. |
| `segformer_b2_generalization_probe.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | at most 15 epochs / 48,000 steps, with patience 5 | cross-entropy | Test whether gentler encoder adaptation improves early held-out behavior and measure a deterministic train/development hard-metric gap. |
| `segformer_b2_generalization_probe_intermediate_lr.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | at most 15 epochs / 48,000 steps, with patience 5 | cross-entropy | Controlled next comparison: change only encoder LR from `6e-6` to `2e-5` while keeping the stable probe protocol fixed. |
| `segformer_b2_generalization_probe_ce_lovasz.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | at most 15 epochs / 48,000 steps, with patience 5 | `0.8 CE + 0.2 Lovasz-Softmax` | Controlled IoU-aligned follow-up: retain the stable `6e-6` encoder LR and add a modest known-class Lovasz term. |
| `segformer_b2_generalization_probe_ce_lovasz_smoke.yaml` | 256 from `train-v1` | 64 from `development-v1` | 2 epochs / at most 256 steps | `0.8 CE + 0.2 Lovasz-Softmax` | Local integration check for the new loss, gradients, checkpoints, and component-aware artifacts; not performance evidence. |

Smoke runs are integration tests, not recipe-selection evidence. Their small
subsets give noisy and biased metrics. The old `baseline`,
`moderately_balanced`, and corresponding smoke files remain checked historical
controls, but they are not the next recommended commands. The completed
comparison found no practically meaningful held-out mIoU gain from
inverse-square-root weighting and worse results on several high-support and
ObjectNav-relevant measures. The new recipe therefore retains ordinary
per-pixel cross-entropy and corrects the training pipeline.

The 48,000-step `generalization_probe` is now the stable B2 reference: it kept
development cross-entropy near `0.82` while matching the historical run's hard
segmentation metrics within a few tenths of a point. Do not extend that run to
160,000 steps. The checked `intermediate_lr` follow-up changes exactly one
optimization variable and asks whether more encoder adaptation can improve
mIoU and pixel accuracy without restoring the old development-loss divergence.

The subsequent `ce_lovasz` probe returns to the stable `6e-6` encoder LR and
changes only the objective. Its cross-entropy term is the same unweighted
41-way per-pixel loss used by the stable probe. Lovasz-Softmax is macro-averaged
over ground-truth-present known classes 1-40 in each minibatch at native decoder
resolution. Unknown class 0 remains a valid negative for those known classes;
target 255 remains ignored. This is a controlled attempt to move argmax IoU,
not permission to replace the development protocol or inspect official val.

The historical 160,000-step reference follows the official SegFormer ADE20K configuration in
the parts portable to this project:

- fit-preserving random resize inside `(2048, 512)` with scale ratio `0.5--2.0`;
- `512 x 512` category-ratio-aware crop, paired horizontal flip, RGB-only
  photometric distortion, ImageNet normalization, and target-255 padding;
- AdamW at `6e-5` for pretrained parameters and `6e-4` for the complete decode
  head, with no weight decay on one-dimensional normalization/bias parameters;
- linear 1,500-step warm-up from `1e-6` of base LR, followed by iteration-based
  linear polynomial decay to zero at 160,000 optimizer steps;
- a server batch of 16 to use the available 96-GiB single GPU and expose about
  2.56 million augmented samples over the capped run.

This is an adaptation, not a claim of bit-for-bit reproduction: HM3D/MPCAT40,
the ADE-pretrained starting checkpoint, a single GPU, and this evaluator differ
from the original ADE20K experiment. The exact upstream configuration is
[NVLabs' SegFormer-B2 ADE20K recipe](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py),
with its separate [ADE20K data pipeline](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py).

In this table, **all `train-v1` does not mean `train-all-v1`**. The names refer
to different dataset roots and different protocol stages:

| Stage | Model input | Role |
|---|---|---|
| Recipe development | `train-v1` (130 scenes) | Update weights for the recommended candidate. |
| Recipe selection | `development-v1` (15 disjoint scenes) | Compare held-out metrics and choose the recipe and training duration. It never supplies gradients. |
| Final refit | `train-all-v1` (all 145 training scenes) | Refit the frozen recipe once using all official training scenes. This is a fresh render, not a concatenation of the other directories. |
| Final evaluation | `official-val-v1` (36 official validation scenes) | Evaluate the frozen final checkpoint once. It never supplies gradients or selects the recipe. |

After the new recipe run is accepted, create and commit a distinct checked
`segformer_b2_final.yaml` based on it. It must use
`train-all-v1`, set `development: null`, remove subset limits, use a distinct
run name, and freeze the preselected duration. The file is deliberately not
created before development selects that duration. The final refit uses
`checkpoints/last`.

`official-val-v1` is absent from every training YAML: it is passed only to the
later explicit `evaluate` command. The exact protocol is documented in
[server training](../../docs/08_server_training.md) and
[server evaluation and calibration](../../docs/09_server_evaluation_and_calibration.md).

## Configuration contract

- `training.datasets` contains portable dataset directory names. They resolve
  below the host-specific `paths.generated_data_root`; the saved
  `provenance/resolved_config.yaml` records the resulting absolute paths.
- `development: null` disables development evaluation. This is explicit in
  tiny-overfit and cannot be accidentally re-enabled by `configs/local.yaml`.
- `max_train_samples` and `max_development_samples` are editable positive
  limits. `null` means the complete manifest.
- Limited train and development sets use their explicit selection strategies
  and seed. Exact IDs are preserved in `provenance/provenance.json`.
- `deterministic_algorithms: false` keeps the normal high-performance CUDA
  kernels while preserving deterministic sample selection and augmentation.
  Set it to `true` only for a strict same-host reproducibility run; PyTorch then
  rejects unsupported nondeterministic operations instead of silently using
  them. The resolved policy is stored in run provenance.
- `evaluate_train_subset` requests an unaugmented hard-metric diagnostic after
  training. `train_subset_evaluation_samples` limits that diagnostic without
  limiting the data used to update weights. Tiny overfit instead leaves the
  diagnostic limit null and evaluates its complete four-image training subset.
- `save_min_development_loss_checkpoint: true` preserves
  `checkpoints/min_development_loss` in addition to the normal highest-mIoU
  `checkpoints/best` and final `checkpoints/last`. It does not change early
  stopping or primary checkpoint selection.
- `qualitative_samples: 10` fixes up to ten ground-truth-selected views per
  active split. Selection is deterministic, scene-diverse, biased toward broad
  and rare class coverage, and never uses predictions. Tiny overfit therefore
  uses all four selected images.
- `qualitative_every_epochs: 1` records those fixed train/development views at
  every epoch. Increase it only when deliberately trading temporal resolution
  for less diagnostic I/O; development capture reuses the existing evaluation
  pass, while train capture adds at most ten unaugmented forward passes.
- `class_weighting: inverse_sqrt` computes weights only from the selected
  training pixels. It remains only for reproducing the historical weighted
  ablation; the recommended recipe sets `class_weighting: none`.
- `max_optimizer_steps` makes the polynomial schedule iteration-based. `epochs`
  remains a hard safety cap; training stops when either limit is reached.
- `head_learning_rate` applies the official ten-times multiplier to the complete
  SegFormer decode head. Older configs without this key retain their legacy
  classifier-only fallback.
- `evaluation` controls held-out metrics. `bootstrap_samples` and
  `calibration_bins` are secondary statistical reporting settings; they do not
  fit a temperature or affect checkpoint selection. `qualitative_samples: 10`
  controls fixed views captured during the existing inference pass.
- `resume: null`, `class_weights: null`, and
  `early_stopping_patience: null` state that those optional behaviors are off.

## Command boundaries

```text
train      update SegFormer weights; optionally evaluate development data
evaluate   freeze weights; compute metrics on an explicit dataset
calibrate  freeze weights; fit one temperature on calibration-fit-v1
evaluate   freeze everything; measure calibrated probabilities on
           calibration-evaluation-v1
```

Calibration is deliberately deferred until the final hard-segmentation model
is accepted; see [server evaluation and calibration](../../docs/09_server_evaluation_and_calibration.md).

## Run artifacts

Every command writes a new collision-safe directory below the ignored,
host-specific `paths.runs_root`, normally:

```text
~/hm3d-semseg-data/runs/<run_name>/
```

Concrete current examples are
`/home/joaocb2002/hm3d-semseg-data/runs/overfit_tiny/` on the workstation and
`/workspace/runs/segformer_b2_ade20k_recipe/` on `knuth`. If a name was
already occupied, use the exact collision-safe suffixed directory printed by
training.

If that name exists, the allocator uses `-002`, `-003`, and so on. It never
mixes a fresh run with previous artifacts. See
[losses, metrics, model selection, and artifacts](../../docs/reference_losses_metrics_and_artifacts.md) for the
contents of each run directory.

The primary human entry point after a run is
`<actual_run_directory>/index.html`. The report is generated
automatically and can be safely rebuilt from raw artifacts with
`hm3d-semseg report-run --run <actual_run_directory>`.
