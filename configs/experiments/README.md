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
| `segformer_b2_baseline_smoke.yaml` | 512 from `train-v1` | 256 from `development-v1` | 10 | none | Exercise baseline training and held-out metrics locally. |
| `segformer_b2_moderately_balanced_smoke.yaml` | 512 from `train-v1` | 256 from `development-v1` | 10 | inverse square root | Exercise class-weight computation and the balanced path locally. |
| `segformer_b2_baseline.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | 20 | none | Full recipe-development candidate on the server. |
| `segformer_b2_moderately_balanced.yaml` | all of 130-scene `train-v1` | all of 15-scene `development-v1` | 20 | inverse square root | Full recipe-development candidate on the server. |

The smoke runs are integration tests, not evidence for choosing the final
recipe. Their small subsets can give noisy and biased metrics.

In this table, **all `train-v1` does not mean `train-all-v1`**. The names refer
to different dataset roots and different protocol stages:

| Stage | Model input | Role |
|---|---|---|
| Recipe development | `train-v1` (130 scenes) | Update weights for the baseline and balanced candidates. |
| Recipe selection | `development-v1` (15 disjoint scenes) | Compare held-out metrics and choose the recipe and training duration. It never supplies gradients. |
| Final refit | `train-all-v1` (all 145 training scenes) | Refit the frozen recipe once using all official training scenes. This is a fresh render, not a concatenation of the other directories. |
| Final evaluation | `official-val-v1` (36 official validation scenes) | Evaluate the frozen final checkpoint once. It never supplies gradients or selects the recipe. |

The final-refit invocation deliberately overrides the selected recipe's
`training.datasets.train` with `train-all-v1`, sets
`training.datasets.development: null`, and uses `checkpoints/last` after the
preselected duration. `official-val-v1` is deliberately absent from every
training YAML: it is passed only to the later explicit `evaluate` command.
The exact final-refit and evaluation commands are documented in
[training](../../docs/07_training.md) and
[evaluation and calibration](../../docs/08_evaluation_and_calibration.md).

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
- `class_weighting: inverse_sqrt` computes weights only from the selected
  training pixels for a smoke run and from the complete training manifest for
  a full run.
- `evaluation` controls held-out metrics. `bootstrap_samples` and
  `calibration_bins` are statistical reporting settings; they do not fit a
  temperature.
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
see [evaluation and calibration](../../docs/08_evaluation_and_calibration.md).

## Run artifacts

Every command writes a new collision-safe directory below the ignored,
host-specific `paths.runs_root`, normally:

```text
~/hm3d-semseg-data/runs/<run_name>/
```

If that name exists, the allocator uses `-002`, `-003`, and so on. It never
mixes a fresh run with previous artifacts. See
[losses, metrics, and run artifacts](../../docs/losses_and_metrics.md) for the
contents of each run directory.
