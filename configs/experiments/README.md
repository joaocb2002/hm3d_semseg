# Experiment configurations

These YAML files configure model-weight training and optional development
evaluation. They are grouped by base model and intended execution host:

```text
configs/experiments/
├── segformer-b2-workstation/  # tiny overfit and bounded smoke runs
├── segformer-b2-server/       # full-data B2 recipes and controlled probes
└── segformer-b5-server/       # full-data B5 probes matched to the B2 probes
```

Moving a YAML file into this layout changes only its command-line path. Model
revision, data, seed, optimization, augmentation, evaluation, and run names are
unchanged. Immutable pretrained revisions are now pinned in each checked
experiment rather than in host-local configuration. `hm3d-semseg train` never
fits a calibration temperature.

The separate `hm3d-semseg smoke-test` command is a render-to-inference systems
check implemented in Python; it is not a training recipe and does not consume
an experiment YAML. Files ending in `_smoke.yaml` are bounded training
experiments over already-generated datasets.

## B2 workstation diagnostics

| Config | Train data | Development data | Duration | Loss | Purpose |
|---|---:|---:|---:|---|---|
| `segformer-b2-workstation/overfit_tiny.yaml` | 4 from `train-v1` | `null` | 50 epochs | cross-entropy | Verify memorization and training mechanics. |
| `segformer-b2-workstation/baseline_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 epochs / 5,120 steps | cross-entropy | Exercise the historical baseline path locally. |
| `segformer-b2-workstation/moderately_balanced_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 epochs / 5,120 steps | inverse-square-root-weighted cross-entropy | Exercise the historical class-weighted path locally. |
| `segformer-b2-workstation/ade20k_recipe_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 epochs / 5,120 steps | cross-entropy | Exercise the spatial, augmentation, full-head, and polynomial-schedule path locally. |
| `segformer-b2-workstation/generalization_probe_ce_lovasz_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 epochs / 5,120 steps | `0.8 CE + 0.2 Lovasz-Softmax` | Exercise the mixed-loss path and component-aware artifacts locally. |

## B2 server recipes

| Config | Duration | Loss or controlled change | Purpose |
|---|---:|---|---|
| `segformer-b2-server/baseline.yaml` | 20 epochs | historical CE pipeline | Preserve the original full-data control. |
| `segformer-b2-server/moderately_balanced.yaml` | 20 epochs | historical inverse-square-root CE | Preserve the original paired class-weighting control. |
| `segformer-b2-server/ade20k_recipe.yaml` | at most 160,000 steps | historical ADE-style CE pipeline | Preserve the completed long reference; do not run by default. |
| `segformer-b2-server/generalization_probe.yaml` | at most 15 epochs / 48,000 steps, patience 5 | stable CE reference | Measure full-data held-out behavior and the train/development gap. |
| `segformer-b2-server/generalization_probe_inverse_sqrt.yaml` | same | inverse-square-root CE | Change only class weighting and run name from the stable probe. |
| `segformer-b2-server/generalization_probe_intermediate_lr.yaml` | same | encoder LR `6e-6` to `2e-5` | Change only encoder LR and run name from the stable probe. |
| `segformer-b2-server/generalization_probe_ce_lovasz.yaml` | same | `0.8 CE + 0.2 Lovasz-Softmax` | Change only the loss and run name from the stable probe. |

Every B2 server recipe uses all 51,215 frames from the 130-scene `train-v1`
and evaluates all 3,729 frames from the disjoint 15-scene `development-v1`.

## B5 server probes

The four files under `segformer-b5-server/` correspond one-to-one with the B2
`generalization_probe` files. Each B5 probe changes only `model_id`, pinned
model revision, and `run_name` from its B2 counterpart. In particular, it keeps
the same 512-pixel crop pipeline rather than introducing 640-pixel crops, so
the first comparison isolates model capacity. B5 was ADE-finetuned at 640 x
640 and accepts the checked 512 x 512 training crops.

## Effective training augmentation

Augmentation is applied online when a training sample is loaded; it does not
create additional files in a generated dataset. The random sequence is
reproducible from the run seed, epoch, and sample index. The checked recipes
use these three regimes:

| Recipes | Training-time behavior |
|---|---|
| `segformer-b2-workstation/overfit_tiny.yaml` | No random geometry, photometry, blur, or sensor noise. This is intentional: the four views must be memorized exactly. |
| B2 `baseline`, `moderately_balanced`, and their workstation smoke variants | Preserve native shape; paired horizontal flip with probability 0.5; RGB-only brightness/contrast/color jitter of magnitude 0.1; RGB-only Gaussian blur with probability 0.05 and radius 0.5; RGB-only Gaussian sensor noise with standard deviation `0.01 * 255`. |
| B2 `ade20k_recipe`, every B2 `generalization_probe*`, their corresponding workstation smoke variants, and every B5 probe | Fit the aligned RGB/mask pair inside a `2048 x 512` box at a uniformly sampled scale from 0.5 to 2.0; take a paired `512 x 512` crop, trying up to ten times to keep the largest valid class below 75% of the crop; paired horizontal flip with probability 0.5; ADE-style RGB-only randomized brightness, contrast, saturation, and hue; pad bottom/right to `512 x 512` when needed. |

Every geometric operation is identical for RGB and mask; RGB resize is
bilinear, mask resize is nearest-neighbor, and padded mask pixels are target
255 (`ignore_index`). Photometric transforms affect RGB only. ImageNet
normalization is always applied after the optional augmentation, but it is a
deterministic model-input conversion rather than dataset augmentation.

Development evaluation, explicit `evaluate` runs, train-subset diagnostics,
and qualitative snapshots use `augment: false`: they preserve each stored
sample's native geometry and apply only ImageNet normalization. Thus metrics
are never computed on randomly augmented development images.

There are four checked `_smoke.yaml` training recipes. Together with
`overfit_tiny.yaml`, they make five workstation training diagnostics. Every
smoke recipe uses 1,024/256 scene-diverse samples, batch 2, ten complete epochs,
and explicitly disables early stopping. A smoke run may still stop on an
exception or manual interruption, but not because development metrics stall.

Smoke runs are integration tests, not recipe-selection evidence. Their small
subsets give noisy and biased metrics. The old `baseline`,
`moderately_balanced`, and corresponding smoke files remain checked historical
controls. Run all four smoke files for an exhaustive workstation regression,
but do not use those subset results to choose a server recipe. The completed
full-data comparison found no practically meaningful held-out mIoU gain from
inverse-square-root weighting and worse results on several high-support and
ObjectNav-relevant measures. The new recipe therefore retains ordinary
per-pixel cross-entropy and corrects the training pipeline.

The 48,000-step `generalization_probe` is now the stable B2 reference: it kept
development cross-entropy near `0.82` while matching the historical run's hard
segmentation metrics within a few tenths of a point. Do not extend that run to
160,000 steps. The checked `intermediate_lr` follow-up changes exactly one
optimization variable and asks whether more encoder adaptation can improve
mIoU and pixel accuracy without restoring the old development-loss divergence.

The `inverse_sqrt` probe is the contemporary controlled class-weighting
comparison. It keeps the stable probe's data, seed, augmentation, parameter
groups, learning rates, schedule, stopping rules, and evaluation unchanged;
only `class_weighting` and the collision-safe run name differ. The older
`baseline` and `moderately_balanced` pair remain historical controls so their
completed runs can still be reproduced exactly.

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

After a model and recipe are accepted, create and commit a distinct checked
`segformer-b2-server/final.yaml` or `segformer-b5-server/final.yaml` based on
the winner. It must use
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
  training pixels. It appears in historical controls and one modern controlled
  probe; the stable reference sets `class_weighting: none`.
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
