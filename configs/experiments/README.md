# Experiment configurations

These YAML files configure model-weight training and optional development
evaluation. `hm3d-semseg train` never fits a calibration temperature.

The separate `hm3d-semseg smoke-test` command is a render-to-inference systems
check implemented in Python; it is not a training recipe and does not consume
an experiment YAML. The two files ending in `_smoke.yaml` below are bounded
training experiments over already-generated datasets.

## Planned sequence

| Config | Train data | Development data | Epochs | Weighting | Purpose |
|---|---:|---:|---:|---|---|
| `overfit_tiny.yaml` | 4 from `train-v1` | `null` | 50 | none | Verify memorization and training mechanics. |
| `segformer_b2_baseline_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 | none | Exercise baseline training and held-out metrics locally. |
| `segformer_b2_moderately_balanced_smoke.yaml` | 1,024 from `train-v1` | 256 from `development-v1` | 10 | inverse square root | Exercise class-weight computation and the balanced path locally. |
| `segformer_b2_baseline.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | 20 | none | Full recipe-development candidate on the server. |
| `segformer_b2_moderately_balanced.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | 20 | inverse square root | Full recipe-development candidate on the server. |

The smoke runs are integration tests, not evidence for choosing the final
recipe. Their small subsets can give noisy and biased metrics. The 1,024-image
training limit gives the scene-diverse selector more views while remaining a
bounded workstation acceptance run; it approximately doubles the training
portion relative to the earlier 512-image smoke recipe.

In this table, **all `train-v1` does not mean `train-all-v1`**. The names refer
to different dataset roots and different protocol stages:

| Stage | Model input | Role |
|---|---|---|
| Recipe development | `train-v1` (130 scenes) | Update weights for the baseline and balanced candidates. |
| Recipe selection | `development-v1` (15 disjoint scenes) | Compare held-out metrics and choose the recipe and training duration. It never supplies gradients. |
| Final refit | `train-all-v1` (all 145 training scenes) | Refit the frozen recipe once using all official training scenes. This is a fresh render, not a concatenation of the other directories. |
| Final evaluation | `official-val-v1` (36 official validation scenes) | Evaluate the frozen final checkpoint once. It never supplies gradients or selects the recipe. |

After recipe selection, create and commit a distinct checked
`segformer_b2_final.yaml` based on the winning candidate. It must use
`train-all-v1`, set `development: null`, remove subset limits, use a distinct
run name, and freeze the preselected duration. The file is deliberately not
created before the baseline/balanced comparison because its weighting and epoch
count are not known yet. The final refit uses `checkpoints/last`.

`official-val-v1` is absent from every training YAML: it is passed only to the
later explicit `evaluate` command. The exact protocol is documented in
[server training](../../docs/08_server_training.md) and
[server evaluation and calibration](../../docs/09_server_evaluation_and_calibration.md).

## Configuration contract

- `training.datasets` contains portable dataset directory names. They resolve
  below the host-specific `paths.generated_data_root`; the saved
  `resolved_config.yaml` records the resulting absolute paths.
- `development: null` disables development evaluation. This is explicit in
  tiny-overfit and cannot be accidentally re-enabled by `configs/local.yaml`.
- `max_train_samples` and `max_development_samples` are editable positive
  limits. `null` means the complete manifest.
- Limited train and development sets use their explicit selection strategies
  and seed. Exact IDs are preserved in `provenance.json`.
- `deterministic_algorithms: false` keeps the normal high-performance CUDA
  kernels while preserving deterministic sample selection and augmentation.
  Set it to `true` only for a strict same-host reproducibility run; PyTorch then
  rejects unsupported nondeterministic operations instead of silently using
  them. The resolved policy is stored in run provenance.
- `evaluate_train_subset` is used only for the tiny memorization report.
- `qualitative_samples: 10` fixes up to ten ground-truth-selected views per
  active split. Selection is deterministic, scene-diverse, biased toward broad
  and rare class coverage, and never uses predictions. Tiny overfit therefore
  uses all four selected images.
- `qualitative_every_epochs: 1` records those fixed train/development views at
  every epoch. Increase it only when deliberately trading temporal resolution
  for less diagnostic I/O; development capture reuses the existing evaluation
  pass, while train capture adds at most ten unaugmented forward passes.
- `class_weighting: inverse_sqrt` computes weights only from the selected
  training pixels for a smoke run and from the complete training manifest for
  a full run.
- `evaluation` controls held-out metrics. `bootstrap_samples` and
  `calibration_bins` are statistical reporting settings; they do not fit a
  temperature. `qualitative_samples: 10` controls the fixed views captured by
  an explicit later `evaluate` command during its existing inference pass.
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

Calibration is performed only after the final server-trained model is frozen;
see [server evaluation and calibration](../../docs/09_server_evaluation_and_calibration.md).

## Run artifacts

Every command writes a new collision-safe directory below the ignored,
host-specific `paths.runs_root`, normally:

```text
~/hm3d-semseg-data/runs/<run_name>/
```

Concrete current examples are
`/home/joaocb2002/hm3d-semseg-data/runs/overfit_tiny/` on the workstation and
`/workspace/runs/segformer_b2_baseline/` plus
`/workspace/runs/segformer_b2_moderately_balanced/` on `knuth`. If a name was
already occupied, use the exact collision-safe suffixed directory printed by
training.

If that name exists, the allocator uses `-002`, `-003`, and so on. It never
mixes a fresh run with previous artifacts. See
[losses, metrics, model selection, and artifacts](../../docs/reference_losses_metrics_and_artifacts.md) for the
contents of each run directory.

The primary human entry point after a run is
`<actual_run_directory>/report/index.html`. The report is generated
automatically and can be safely rebuilt from raw artifacts with
`hm3d-semseg report-run --run <actual_run_directory>`.
