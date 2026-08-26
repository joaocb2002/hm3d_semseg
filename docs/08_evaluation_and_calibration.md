# 8. Evaluation and calibration

**Execution location: GPU server.** Enter this step with the frozen final
checkpoint and the previously transferred official/calibration datasets. Exit
it with a complete final run containing official evaluation, disjoint
calibration evaluation, benchmark results, and `checkpoints/calibrated`.

These commands run on the GPU server that produced the checkpoint.
Official validation remains untouched until the development recipe and duration
are frozen. After calibration and benchmarking, return the complete final run
and calibrated checkpoint using the
[workstation-to-server handoff guide](06a_server_handoff.md).

Calibration is not part of any `train` run, including tiny overfit and the two
local smoke trials. Only after final training does `hm3d-semseg calibrate`
freeze all SegFormer weights and optimize one scalar temperature on
`calibration-fit-v1`; the following `evaluate` command measures probability
quality on the disjoint `calibration-evaluation-v1` root.

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

Step 06a prepared and transferred `official-val-v1`, `calibration-fit-v1`, and
`calibration-evaluation-v1`; it did not evaluate them. Do not regenerate these
on the training-only server. Use the official 36-scene validation only after
freezing the recipe and duration:

```bash
hm3d-semseg evaluate \
  --checkpoint /absolute/final/checkpoints/last \
  --dataset /absolute/server/data/official-val-v1 \
  --output /absolute/final/evaluation-official-val \
  --config /absolute/path/to/frozen-final-experiment.yaml \
  --local-config configs/local.yaml
```

Call these validation results, never private-test results.

The checked-in calibration lists partition the 36 official validation scenes
into 12 temperature-fit and 24 probability-evaluation scenes. Freeze model
weights and fit only the scalar temperature on the former:

```bash
hm3d-semseg calibrate \
  --checkpoint /absolute/final/checkpoints/last \
  --dataset /absolute/server/data/calibration-fit-v1 \
  --output /absolute/final/checkpoints/calibrated \
  --local-config configs/local.yaml
```

`calibration.json` records the fitted value, every fit scene, and optimization
details. Read that value and evaluate calibrated probabilities only on the
disjoint calibration-evaluation scenes:

```bash
CALIBRATED=/absolute/final/checkpoints/calibrated
TEMPERATURE="$(jq -r .temperature "$CALIBRATED/calibration.json")"

hm3d-semseg evaluate \
  --checkpoint "$CALIBRATED" \
  --dataset /absolute/server/data/calibration-evaluation-v1 \
  --output /absolute/final/evaluation-calibration \
  --config /absolute/path/to/frozen-final-experiment.yaml \
  --local-config configs/local.yaml \
  --temperature "$TEMPERATURE"
```

Evaluation rejects overlap with recorded fit scenes. Temperature scaling
preserves argmax, so segmentation metrics can still be reported over all
official validation; label calibrated probability metrics by their disjoint
24-scene evaluation subset.

For final research reporting run at least three seeds and report mean/std. Add
native-resolution batch-1 latency, FPS, p95, peak memory, parameter count,
hardware, precision, warm-up, and timing iterations.

```bash
hm3d-semseg benchmark-inference \
  --checkpoint /absolute/final/checkpoints/calibrated \
  --output /absolute/final/benchmark \
  --warmup 20 \
  --iterations 100
```

## Return gate

Training work on the server is now complete. Before step 09:

1. Checksum the calibrated deployment files.
2. Copy the complete final run and required recipe-development evidence back to
   the workstation with `rsync` or managed artifact storage.
3. Verify the checksums on the workstation.
4. Keep the server copy until local inference and backup succeed.
5. Do not `git add` the returned checkpoint. The workstation source repository
   stays at the commit recorded in `provenance.json`; the run belongs under the
   external `hm3d-semseg-data/runs` tree.

Use the exact commands in
[step 06a, Preserve and return the results](06a_server_handoff.md#9-preserve-and-return-the-results).

Back on the workstation, next: [inference and ObjectNav](09_inference_and_objectnav_integration.md).
