# 9. Server evaluation and calibration

**Execution location: GPU server.** Enter with the frozen final experiment and
its `checkpoints/last`. This stage never changes SegFormer weights. It reports
hard segmentation once on official validation, fits one positive temperature
on dedicated calibration-fit scenes, measures probability quality on disjoint
calibration-evaluation scenes, and benchmarks the deployment checkpoint.

Generic paths below are followed by the planned current paths. If training
allocated a numeric suffix, replace `segformer_b2_final` with the exact run
directory printed by training.

## 9.1 Freeze the inputs before looking at official validation

Confirm that the final run records:

- the selected recipe and frozen epoch count;
- `train-all-v1` as training data and `development: null`;
- the expected Git SHA, model revision, camera hash, dataset manifest, seed,
  package stack, hardware, and one visible GPU;
- finite training values and a complete `checkpoints/last`.

Do not use official-validation results to change the recipe, duration, class
weights, augmentation, or optimizer. Any such change would make official
validation another development set.

## 9.2 Evaluate official hard segmentation once

Generic command:

```bash
hm3d-semseg evaluate \
  --checkpoint /server/runs/FINAL_RUN/checkpoints/last \
  --dataset /server/data/official-val-v1 \
  --output /server/runs/FINAL_RUN/evaluation-official-val \
  --config /server/repository/hm3d-semseg/configs/experiments/FINAL_EXPERIMENT.yaml \
  --local-config configs/local.yaml
```

Open `EVALUATION_OUTPUT/report/index.html` after each explicit evaluation. For
the current final-run layout, the uncalibrated official report will be:

```text
/workspace/runs/segformer_b2_final/evaluation-official-val/report/index.html
```

The report, CSV tables, plots, and fixed ten-view qualitative sheet are derived
from the same streamed evaluation that writes `summary.json`; they do not
change weights or probabilities.

Planned current command:

```bash
cd /workspace/repository/hm3d-semseg

hm3d-semseg evaluate \
  --checkpoint /workspace/runs/segformer_b2_final/checkpoints/last \
  --dataset /workspace/data/official-val-v1 \
  --output /workspace/runs/segformer_b2_final/evaluation-official-val \
  --config /workspace/repository/hm3d-semseg/configs/experiments/segformer_b2_final.yaml \
  --local-config configs/local.yaml
```

Call these **official validation** results, not private-test results. The primary
hard-label result is known-class mIoU over supported MPCAT40 classes 1--40.
Also retain per-class/ObjectNav-six IoU, confusion matrices, overall accuracy,
scene-macro mean/median/bootstrap interval, and efficiency provenance. The
evaluator accumulates one global pixel confusion matrix; it does not average
batch IoUs.

## 9.3 Fit one scalar temperature

Freeze all model weights and fit temperature only on the 12 scenes in
`calibration-fit-v1`.

Generic command:

```bash
hm3d-semseg calibrate \
  --checkpoint /server/runs/FINAL_RUN/checkpoints/last \
  --dataset /server/data/calibration-fit-v1 \
  --output /server/runs/FINAL_RUN/checkpoints/calibrated \
  --local-config configs/local.yaml
```

Planned current command:

```bash
hm3d-semseg calibrate \
  --checkpoint /workspace/runs/segformer_b2_final/checkpoints/last \
  --dataset /workspace/data/calibration-fit-v1 \
  --output /workspace/runs/segformer_b2_final/checkpoints/calibrated \
  --local-config configs/local.yaml
```

`calibration.json` records the fitted positive temperature, fit scenes,
optimizer settings, and first/last recorded optimization-batch NLL. Temperature
divides all 41 logits before softmax. It changes probability sharpness but preserves their
ordering, so the argmax segmentation mask and all hard-label metrics remain
unchanged.

## 9.4 Evaluate probability calibration without leakage

Read the fitted temperature:

```bash
jq -r .temperature \
  /workspace/runs/segformer_b2_final/checkpoints/calibrated/calibration.json
```

Insert that printed scalar into the following generic command:

```bash
hm3d-semseg evaluate \
  --checkpoint /server/runs/FINAL_RUN/checkpoints/calibrated \
  --dataset /server/data/calibration-evaluation-v1 \
  --output /server/runs/FINAL_RUN/evaluation-calibration \
  --config /server/repository/hm3d-semseg/configs/experiments/FINAL_EXPERIMENT.yaml \
  --local-config configs/local.yaml \
  --temperature FITTED_TEMPERATURE
```

Planned current form, replacing only `FITTED_TEMPERATURE` with the printed
number:

```bash
hm3d-semseg evaluate \
  --checkpoint /workspace/runs/segformer_b2_final/checkpoints/calibrated \
  --dataset /workspace/data/calibration-evaluation-v1 \
  --output /workspace/runs/segformer_b2_final/evaluation-calibration \
  --config /workspace/repository/hm3d-semseg/configs/experiments/segformer_b2_final.yaml \
  --local-config configs/local.yaml \
  --temperature FITTED_TEMPERATURE
```

The 24 calibration-evaluation scenes are disjoint from the 12 temperature-fit
scenes. The evaluator rejects scene overlap recorded by calibration provenance.
Report NLL, multiclass Brier score, ECE/reliability, entropy, and risk-coverage
as calibration results for this 24-scene subset. Do not present them as if
estimated on the full 36-scene official-validation root.

## 9.5 Benchmark the deployment checkpoint

Generic command:

```bash
hm3d-semseg benchmark-inference \
  --checkpoint /server/runs/FINAL_RUN/checkpoints/calibrated \
  --output /server/runs/FINAL_RUN/benchmark \
  --warmup 20 \
  --iterations 100
```

Planned current command:

```bash
hm3d-semseg benchmark-inference \
  --checkpoint /workspace/runs/segformer_b2_final/checkpoints/calibrated \
  --output /workspace/runs/segformer_b2_final/benchmark \
  --warmup 20 \
  --iterations 100
```

Preserve native-resolution batch-1 latency, FPS, p95, peak memory, parameter
count, GPU, precision, warm-up, and timing iterations. Server throughput is not
a substitute for later benchmarking on the ObjectNav deployment GPU.

## 9.6 Seal the final run for transfer

Generic checksum procedure:

```bash
cd /server/runs/FINAL_RUN
find checkpoints/calibrated -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > calibrated.sha256
sha256sum -c calibrated.sha256
```

Planned current procedure:

```bash
cd /workspace/runs/segformer_b2_final
find checkpoints/calibrated -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > calibrated.sha256
sha256sum -c calibrated.sha256
```

Keep the complete final run: resolved configuration, provenance, raw and
summarized metrics, evaluation outputs, plots, diagnostics, TensorBoard,
benchmark, best/last/calibrated checkpoints, and checksum manifest. A loose
`model.safetensors` is not a deployable or reproducible checkpoint.

Detailed definitions and output paths are in the
[losses and metrics reference](reference_losses_metrics_and_artifacts.md).

Next: [return the complete run and integrate it with ObjectNav](10_return_and_objectnav.md).
