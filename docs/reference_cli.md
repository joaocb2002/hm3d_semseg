# Reference: CLI

`hm3d-semseg COMMAND --help` is the executable authority.

| Command | Required inputs | Primary output |
|---|---|---|
| `install-training-env` | project root defaults to current directory | Host/profile plan; optionally a verified installation |
| `doctor` | `--local-config` | Dependency/path/data/GPU JSON report |
| `resolve-camera` | output plus local/ObjectNav config | Hashed camera YAML and official comparison |
| `inspect-scene` | local config, split, scene ID, output | Aligned images and `report.json` |
| `audit-taxonomy` | local config, split, output | `audit.json` and scene-by-class CSV |
| `generate-dataset` | command config and local config | Resumable dataset, plan, and progress |
| `validate-dataset` | dataset directory | Full contract/decode report and review panels |
| `download-model` | local config and optional model/revision | Pinned cache snapshot metadata |
| `make-dev-split` | train audit and output | Fit/development lists and coverage report |
| `make-calibration-split` | validation audit and output | Disjoint calibration lists |
| `train` | experiment config and local config | Resolved run, metrics, diagnostics, checkpoints |
| `report-run` | one training-run directory | Derived HTML, Markdown, CSV tables, and plots under `report/` |
| `compare-runs` | repeated run directories and output | Held-out comparison, paired scene interval, tables, and plots |
| `evaluate` | checkpoint, dataset, output | Segmentation/probability report and plots |
| `calibrate` | checkpoint, calibration-fit dataset, output | Calibrated checkpoint and temperature provenance |
| `benchmark-inference` | checkpoint and output | Native batch-1 latency/FPS/memory/size report |
| `infer` | checkpoint, RGB image, output | IDs, color, overlay, confidence, entropy, metadata |
| `smoke-test` | local config | Four-frame render/train/evaluate/reload diagnostic |

## Important behavior

- `install-training-env` is a dry plan unless `--apply` is present;
  `--run-tests` adds verification; auto chooses a host-compatible CPU/CUDA
  profile.
- `generate-dataset --dry-run` prints a plan and intentionally reports
  `validation_only: true`; real generation reports `validation_only: false` at
  completion. Progress is enabled unless `--no-progress` is passed.
- `validate-dataset` checks the global manifest contract and fully decodes the
  selected root's RGB/mask files.
- `download-model --revision` should receive an immutable commit. The current
  frozen SegFormer revision is
  `de01bae28967510f9ddd496c60a969357195400c`.
- `train` shows the optimizer-step plan, ETA, loss, learning rates, throughput,
  device, AMP, and parameters. Fresh run-name collisions allocate a numeric
  suffix; only explicit resume appends to an existing run. It automatically
  writes `report/index.html` after completion.
- `report-run` is read-only with respect to raw run artifacts and checkpoints;
  it replaces only derived files below `RUN/report`.
- `compare-runs` validates comparable provenance and compares held-out metrics,
  not differently weighted training objectives.
- A limited training experiment validates its deterministic selected files plus
  the complete manifest/schema/split contract. A full run validates all files.
- `evaluate` requires explicit checkpoint, dataset, and output. It never changes
  model weights. It captures the configured fixed qualitative set during the
  same pass and writes `OUTPUT/report/index.html` plus CSV tables.
- `calibrate` copies a complete checkpoint and fits only one temperature.
- `infer` does not save the full `[41,H,W]` probability tensor unless
  `--save-probabilities` is requested.

## Examples

Dataset pilot:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/pilot.yaml \
  --local-config configs/local.yaml \
  --max-scenes 1 \
  --max-samples-per-scene 4
```

Generic evaluation:

```bash
hm3d-semseg evaluate \
  --checkpoint /server/runs/FINAL_RUN/checkpoints/last \
  --dataset /server/data/official-val-v1 \
  --output /server/runs/FINAL_RUN/evaluation-official-val \
  --config /server/repository/hm3d-semseg/configs/experiments/FINAL_EXPERIMENT.yaml \
  --local-config configs/local.yaml
```

Current workstation smoke-report example:

```bash
hm3d-semseg report-run \
  --run /home/joaocb2002/hm3d-semseg-data/runs/segformer_b2_baseline_smoke
```

Current server candidate-comparison example:

```bash
hm3d-semseg compare-runs \
  --run /workspace/runs/segformer_b2_baseline \
  --run /workspace/runs/segformer_b2_moderately_balanced \
  --output /workspace/runs/comparisons/baseline-vs-balanced
```

Planned current evaluation:

```bash
hm3d-semseg evaluate \
  --checkpoint /workspace/runs/segformer_b2_final/checkpoints/last \
  --dataset /workspace/data/official-val-v1 \
  --output /workspace/runs/segformer_b2_final/evaluation-official-val \
  --config /workspace/repository/hm3d-semseg/configs/experiments/segformer_b2_final.yaml \
  --local-config configs/local.yaml
```

Unknown YAML keys, relative external paths, label counts other than 41, an
overlapping ignore index, unsupported codecs, invalid taxonomy policies, and
scene leakage fail before costly work. Dataset names in experiment YAML resolve
below the absolute `paths.generated_data_root`; `null` limits mean a complete
manifest and `development: null` disables development evaluation.

Return to the [execution guide](README.md).
