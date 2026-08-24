# 12. CLI reference

Use `hm3d-semseg COMMAND --help` as the executable authority.

| Command | Required inputs | Primary output |
|---|---|---|
| `install-training-env` | project root defaults to current directory | dry-run host/profile/commands report; optional verified install |
| `doctor` | `--local-config` | JSON dependency/path/data/GPU report |
| `resolve-camera` | `--output`, local config or `--objectnav-config` | hashed camera YAML and official comparison |
| `inspect-scene` | local config, split, scene ID, output | aligned images and `report.json` |
| `audit-taxonomy` | local config, split, output | `audit.json`, scene-by-class CSV |
| `generate-dataset` | command config, local config | resumable dataset; supports split list, limits, dry run |
| `validate-dataset` | dataset directory | complete validation report |
| `download-model` | local config; model ID default | pinned cache snapshot metadata |
| `make-dev-split` | audit, output | fit/development text lists and coverage |
| `make-calibration-split` | validation audit, output | disjoint calibration-fit/evaluation lists |
| `train` | experiment config, local config | resolved run, metrics, best/last checkpoints |
| `evaluate` | checkpoint, dataset, output | segmentation/calibration summary and confusion NPY |
| `calibrate` | checkpoint, calibration dataset, output | copied calibrated checkpoint and provenance |
| `benchmark-inference` | checkpoint, output | native batch-1 latency/FPS/memory/size report |
| `infer` | checkpoint, image, output | IDs, color, overlay, confidence, entropy, metadata |
| `smoke-test` | local config | four-frame generation/train/evaluate/reload/infer diagnostic |

## Options and defaults

- `install-training-env`: `--profile auto` chooses `cpu`, `cu118`, `cu126`, or
  `cu130` from NVIDIA compute capability and driver support; dry-run is the
  default; `--apply` executes; `--run-tests` adds unit tests; dev dependencies
  are included unless `--without-dev`; `--force-torch` reinstalls even a matching
  runtime; `--allow-unsupported-host` permits explicit cross-host provisioning.
- `doctor`: required `--local-config PATH`.
- `resolve-camera`: required `--output PATH`; optional `--local-config PATH`,
  `--objectnav-config PATH`; `--raw-yaml-fallback` defaults false.
- `inspect-scene`: required `--local-config`, `--scene-id`, and `--output`;
  `--split minival`; `--num-views 4`.
- `audit-taxonomy`: required `--local-config`, `--split`, and `--output`;
  optional `--rendered-dataset`.
- `generate-dataset`: required `--config` and `--local-config`; optional
  `--split-list`; `--max-scenes` and `--max-samples-per-scene` default to YAML;
  `--official-split` overrides the filesystem split and selects a distinct
  `official-<split>-v1` name; `--dataset-name` overrides that name;
  `--dry-run` defaults false. Scene/view progress with elapsed time and ETA is
  enabled by default for real generation; `--no-progress` disables it.
- `validate-dataset`: required `--dataset`; writes its report and eight
  deterministic manual panels under `<dataset>/validation`.
- `download-model`: required `--local-config`; `--model-id` defaults to
  `nvidia/segformer-b2-finetuned-ade-512-512`; `--revision` defaults to remote
  main but is immediately resolved and recorded as an immutable commit.
- `make-dev-split`: required `--audit` and `--output`; optional
  `--local-config`; `--development-scenes 15`.
- `make-calibration-split`: required `--audit` and `--output`; optional
  `--local-config`; `--fit-scenes 12`.
- `train`: required `--config` and `--local-config`; optimizer-step progress,
  elapsed time, ETA, live loss/LR/throughput, setup, and parameter counts are
  shown by default; `--no-progress` disables terminal progress output. Limited
  runs validate only their deterministic selected files plus global dataset
  contracts. Every completed run writes raw `metrics.jsonl`, compact
  `metrics_summary.json`, TensorBoard events, and static loss/LR plus optimization
  plots; development runs add a development loss/mIoU plot and complete
  per-epoch evaluation reports. Fresh run-name collisions allocate a numbered
  directory instead of mixing artifacts.
- `evaluate`: required `--checkpoint`, `--dataset`, and `--output`; optional
  `--config`, `--local-config`, and `--device`; `--temperature 1.0`.
- `calibrate`: required `--checkpoint`, `--dataset`, and `--output`; optional
  config files and `--device`; `--epochs 5`.
- `benchmark-inference`: required `--checkpoint` and `--output`;
  `--iterations 100`; `--warmup 20`; device auto-selects; half precision is used
  on CUDA unless `--float32`.
- `infer`: required `--checkpoint`, `--image`, and `--output`; device
  auto-selects; `--save-probabilities` defaults false.
- `smoke-test`: required `--local-config`; creates/reuses the named diagnostic
  pilot and run roots and never downloads a missing model.

Common examples:

```bash
hm3d-semseg generate-dataset --help
hm3d-semseg generate-dataset \
  --config configs/data/pilot.yaml \
  --local-config configs/local.yaml \
  --max-scenes 1 \
  --max-samples-per-scene 4
```

```bash
hm3d-semseg evaluate \
  --checkpoint /absolute/checkpoint \
  --dataset /absolute/dataset \
  --output /absolute/evaluation \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

Defaults live in the dataclass schema and checked YAML files. Unknown YAML keys,
relative external paths, model label counts other than 41, an overlapping ignore
index, unsupported codecs, and invalid policy values are rejected before costly
work.
