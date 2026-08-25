# 8. Evaluation and calibration

These commands normally run on the GPU server that produced the checkpoint.
Official validation remains untouched until the development recipe and duration
are frozen. After calibration and benchmarking, return the complete final run
and calibrated checkpoint using the
[workstation-to-server handoff guide](server_handoff.md).

Evaluate a fixed scene-disjoint manifest:

```bash
hm3d-semseg evaluate \
  --checkpoint /absolute/run/checkpoints/best \
  --dataset /absolute/development-dataset \
  --output /absolute/run/evaluation-development \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

The global confusion matrix accumulates across all pixels—batch IoUs are never
averaged. Primary known-class mIoU covers supported MPCAT40 classes 1–40 and
excludes unknown. Reports also contain per-class intersection/union,
precision/recall/F1/support, mIoU-41, unknown behavior, mean recall, overall
accuracy, frequency-weighted IoU, raw/normalized confusion, ObjectNav-six,
per-scene mIoU, scene mean/median, and seeded scene-bootstrap confidence bounds.

Probability metrics are streamed: categorical NLL, multiclass Brier, 15-bin ECE,
reliability bins, and correct/incorrect entropy. Evaluation does not retain
full-resolution probability tensors.

See [losses, metrics, and run artifacts](losses_and_metrics.md) for formulas,
intuitive interpretations, reporting priorities, absent-class rules, and the
exact JSON/NPY/plot paths produced by training and evaluation.

Use the official 36-scene validation only after freezing:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val
hm3d-semseg evaluate \
  --checkpoint /absolute/final/checkpoints/last \
  --dataset /absolute/official-val \
  --output /absolute/final/evaluation-official-val
```

Call these validation results, never private-test results.

Split official validation scenes deterministically into calibration-fit and
calibration-evaluation lists. Fit temperature only on the former:

```bash
hm3d-semseg make-calibration-split \
  --local-config configs/local.yaml \
  --audit /absolute/generated/root/audits/val \
  --output configs/data/splits
```

Generate two validation datasets using the saved scene lists. Then:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val \
  --dataset-name calibration-fit-v1 \
  --split-list configs/data/splits/calibration_fit.txt

hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --official-split val \
  --dataset-name calibration-evaluation-v1 \
  --split-list configs/data/splits/calibration_evaluation.txt
```

Then:

```bash
hm3d-semseg calibrate \
  --checkpoint /absolute/final/checkpoints/last \
  --dataset /absolute/calibration-fit \
  --output /absolute/final/checkpoints/calibrated \
  --local-config configs/local.yaml
```

`calibration.json` records every fit scene and optimization details. Evaluate
calibrated probabilities with `--temperature <saved-value>` only on the disjoint
calibration-evaluation scenes; evaluation rejects overlap with recorded fit
scenes. Temperature scaling preserves argmax, so
segmentation metrics can still be reported over all official validation; label
calibration metrics by their disjoint evaluation subset.

For final research reporting run at least three seeds and report mean/std. Add
native-resolution batch-1 latency, FPS, p95, peak memory, parameter count,
hardware, precision, warm-up, and timing iterations.

```bash
hm3d-semseg benchmark-inference \
  --checkpoint /absolute/final/checkpoints/last \
  --output /absolute/final/benchmark \
  --warmup 20 \
  --iterations 100
```

Next: [inference and ObjectNav](09_inference_and_objectnav_integration.md).
