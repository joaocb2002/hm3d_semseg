# 8. Server training: recipe development and final refit

**Execution location: GPU server.** Start only after step 7 accepts the server.
This stage changes SegFormer weights but performs no temperature calibration and
does not use `official-val-v1` labels. It has two distinct phases:

1. compare candidate recipes trained on `train-v1` and selected on the
   scene-disjoint `development-v1`;
2. freeze the winning recipe and duration, then refit it once on
   `train-all-v1` with no development selection.

The current server is `knuth`; source, data, runs, and cache are respectively
under `/workspace/repository/hm3d-semseg`, `/workspace/data`,
`/workspace/runs`, and `/workspace/cache`.

## 8.1 Enter a durable one-GPU session

Generic direct-server session:

```bash
tmux new -s hm3d-training
```

Then, inside the new session:

```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=GPU_INDEX
cd /server/project/root/repository/hm3d-semseg
```

Current `knuth` form when physical GPU 0 is assigned:

```bash
tmux new -s hm3d-training
```

Then, inside that session:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
cd /workspace/repository/hm3d-semseg
```

Detach with `Ctrl-b`, then `d`; list sessions with `tmux ls`; reattach with
`tmux attach -t hm3d-training`. If the site later introduces Slurm/PBS/LSF,
replace direct `tmux` execution with the required one-GPU job allocation while
keeping all commands and artifact roots unchanged.

## 8.2 Know which command changes what

| Command | Updates SegFormer weights | Uses held-out labels | Fits temperature |
|---|---:|---:|---:|
| `train` | yes | only if `training.datasets.development` is non-null | no |
| `evaluate` | no | yes, from the explicit dataset | no |
| `calibrate` | no | only the explicit calibration-fit dataset | yes, one scalar |

The whole SegFormer-B2 model is fine-tuned. AdamW gives pretrained encoder
parameters a lower learning rate and the new 41-class decode head a ten-times
higher rate. Both are multiplied by the same warm-up/cosine schedule. Loss is
per-pixel cross-entropy on raw logits; target 255 is ignored and class 0 remains
learnable. See the [metrics reference](reference_losses_metrics_and_artifacts.md)
for notation, formulas, and artifact locations.

## 8.3 Candidate A: unweighted baseline

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

Current expected first-run directory:

```text
/workspace/runs/segformer_b2_baseline/
```

If that name already exists, training safely allocates `-002`, `-003`, and so
on. Always use the actual directory printed at startup. This candidate gives
every valid pixel equal loss weight.

## 8.4 Candidate B: moderately balanced

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_moderately_balanced.yaml \
  --local-config configs/local.yaml
```

Current expected first-run directory:

```text
/workspace/runs/segformer_b2_moderately_balanced/
```

This candidate changes only the training loss weighting: inverse-square-root
weights are computed from `train-v1` pixel counts, normalized over supported
classes, capped at 5, and recorded with a checksum. Development loss and metrics
remain unweighted, so the two candidates are directly comparable. Do not add
class-aware oversampling to this comparison.

## 8.5 Select the recipe on development evidence

`checkpoints/best` for both candidates is selected by the highest development
known-class mIoU, not by training loss. Development images never provide
gradients. Do not inspect `official-val-v1` while choosing.

Generic compact inspection:

```bash
jq '{best: .development.best_known_class_miou,
     final: .development.final,
     optimization: .training.optimization}' \
  /server/project/root/runs/RUN_NAME/metrics_summary.json
```

Current baseline example:

```bash
jq '{best: .development.best_known_class_miou,
     final: .development.final,
     optimization: .training.optimization}' \
  /workspace/runs/segformer_b2_baseline/metrics_summary.json
```

Current balanced example:

```bash
jq '{best: .development.best_known_class_miou,
     final: .development.final,
     optimization: .training.optimization}' \
  /workspace/runs/segformer_b2_moderately_balanced/metrics_summary.json
```

Use the actual suffixed run directory if applicable. Compare, in this order:

1. best development known-class mIoU;
2. per-class IoU and ObjectNav-six mIoU at that epoch;
3. scene-macro mean, median, and bootstrap interval;
4. confusion patterns and qualitative predictions;
5. development-curve stability rather than a one-epoch spike;
6. finite loss/gradients, memory headroom, and throughput.

Training loss diagnoses optimization but cannot select the model that
generalizes best. A tiny mIoU difference inside broad scene-bootstrap intervals
is weak evidence; prefer the simpler baseline unless balancing gives a clear,
repeatable improvement in the classes that matter.

Each newly completed run automatically builds a static report at
`RUN/report/index.html`. It contains the run card, warnings, checkpoint table,
sortable epoch/per-class tables, robustly aggregated optimization plots,
development curves, and fixed qualitative sets. The JSON/JSONL files remain
the authority; the report is a replaceable presentation layer.

Generic commands to build or refresh reports and compare candidates are:

```bash
hm3d-semseg report-run --run /server/runs/BASELINE_RUN
hm3d-semseg report-run --run /server/runs/BALANCED_RUN

hm3d-semseg compare-runs \
  --run /server/runs/BASELINE_RUN \
  --run /server/runs/BALANCED_RUN \
  --output /server/runs/comparisons/baseline-vs-balanced
```

Current `knuth` form, assuming unsuffixed run names:

```bash
hm3d-semseg report-run \
  --run /workspace/runs/segformer_b2_baseline
hm3d-semseg report-run \
  --run /workspace/runs/segformer_b2_moderately_balanced

hm3d-semseg compare-runs \
  --run /workspace/runs/segformer_b2_baseline \
  --run /workspace/runs/segformer_b2_moderately_balanced \
  --output /workspace/runs/comparisons/baseline-vs-balanced
```

Open `report/index.html` or `comparisons/baseline-vs-balanced/index.html` from
the VS Code Remote SSH file explorer. The comparison checks dataset manifests,
model revision, and seed; compares only held-out metrics; and reports paired
per-scene differences with a bootstrap interval. It deliberately does not
compare baseline and balanced training-loss magnitudes because those optimize
different objectives.

`report-run` can backfill scalar plots and tables for runs created by an older
commit. It cannot reconstruct predictions for epochs whose checkpoints were
not retained. Therefore an already-running older server candidate remains
without the new ten-image development history, but all its existing numerical
metrics remain valid. Runs started from this commit capture the history during
training.

For live curves, start a separate session:

```bash
tmux new -s hm3d-tensorboard
```

Inside that session:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate hm3d-semseg-train
tensorboard --logdir /workspace/runs --host 127.0.0.1 --port 6006
```

In VS Code Remote SSH, forward port 6006 from the **Ports** panel. Static
HTML/PNG/CSV files remain the archival view. TensorBoard adds zooming,
smoothing, exact values, wall time, finite/non-finite optimizer health,
throughput, memory, expanded development metrics, and train/development contact
sheets for the fixed qualitative sets.

## 8.6 Freeze the final recipe and duration

Once the comparison is accepted, write down:

- selected candidate and its run directory;
- selected development epoch and known-class mIoU;
- final epoch count (normally selected zero-based epoch plus one);
- seed, augmentation, batch/accumulation semantics, learning rates, weighting,
  and all environment/source provenance.

Create a distinct checked experiment such as
`configs/experiments/segformer_b2_final.yaml` on the workstation. Base it on
the selected candidate, then change only the final-protocol fields:

```yaml
training:
  datasets:
    train: train-all-v1
    development: null
  max_train_samples: null
  max_development_samples: null
  run_name: segformer_b2_final
  epochs: FROZEN_EPOCH_COUNT
```

That file is **planned**, not present in the repository at the time this guide
was written. Review and commit it on the workstation, push it, then fetch and
check out that new recorded commit on the server before final refitting. Do not
edit an untracked scientific config only inside the server clone.
`FROZEN_EPOCH_COUNT` is intentionally not replaced with a fabricated current
value: derive it only after the running baseline/balanced development comparison
is accepted.

The generic final command is:

```bash
hm3d-semseg train \
  --config configs/experiments/FROZEN_FINAL_EXPERIMENT.yaml \
  --local-config configs/local.yaml
```

Planned current command after `segformer_b2_final.yaml` is committed:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_final.yaml \
  --local-config configs/local.yaml
```

Planned current output root:

```text
/workspace/runs/segformer_b2_final/
```

The final refit receives gradients from all 145 official training scenes. It
has no development dataset and must not tune against official validation.
Therefore `checkpoints/last`, at the frozen duration, is the protocol checkpoint;
`checkpoints/best` in a no-development run merely tracks minimum training epoch
loss and is not evidence of better generalization.

Every run keeps `metrics.jsonl`, `metrics_summary.json`, `summary.json`,
`resolved_config.yaml`, `provenance.json`, TensorBoard events, plots,
diagnostics, the derived `report/`, and atomic best/last checkpoints under
`/workspace/runs`. These artifacts never enter Git.

Next: [evaluate and calibrate the frozen final checkpoint](09_server_evaluation_and_calibration.md).
