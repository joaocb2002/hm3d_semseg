# Losses, metrics, and run artifacts

This document defines the quantities reported by `hm3d-semseg`, explains what
they mean, and identifies the artifact that contains each value. Keep this
protocol fixed when comparing experiments: changing the class set, treatment of
absent classes, scene split, aggregation rule, or calibration temperature changes
the meaning of a result.

## What matters at each stage

| Stage | Primary evidence | Important supporting evidence | What it cannot establish |
|---|---|---|---|
| Tiny overfit | Falling training cross-entropy and near-memorization on the four selected images | Selected-subset pixel accuracy, known-class mIoU, per-class IoU, confusion, qualitative masks | Generalization to a new image or scene |
| Recipe development | Development known-class mIoU; this selects `checkpoints/best` | Development cross-entropy, per-class IoU, ObjectNav-six mIoU, scene-macro mIoU, confusion and calibration metrics | Final unbiased performance after repeated tuning |
| Final fixed evaluation | Known-class mIoU on unseen scenes | Per-class IoU, ObjectNav-six mIoU, scene-macro mean/median/95% interval, pixel accuracy, unknown behavior | Probability reliability unless calibration metrics are also reported |
| Calibrated evaluation | NLL, multiclass Brier score, ECE and risk-coverage on calibration-evaluation scenes | The unchanged hard-label segmentation metrics | Better segmentation regions: positive temperature scaling does not change argmax labels |

Mean average precision, or mAP, is not used. It is principally an object
detection or instance-segmentation metric requiring object instances, matches,
and confidence-ranked detections. This project predicts one semantic class per
pixel without distinguishing object instances.

## Prediction contract

For pixel `p`, SegFormer produces 41 logits `z[p,c]`: learnable class 0
`unknown` and MPCAT40 classes 1--40. Logits are bilinearly upsampled to the
native mask shape. For a positive temperature `T`, probabilities are:

```text
p[p,c] = exp(z[p,c] / T) / sum(exp(z[p,k] / T), k = 0..40)
```

The hard label and confidence are:

```text
predicted_label[p] = argmax(p[p,c], c = 0..40)
confidence[p]      = max(p[p,c], c = 0..40)
```

Softmax preserves ordering, so `argmax(softmax(logits))` and `argmax(logits)`
give the same label. Temperature changes confidence but not the label.

Target 255 is `ignore_index`, not a model class. Such pixels contribute neither
loss nor metrics. Target 0 is a real `unknown` class and does contribute to
training, overall accuracy, and `miou_41`; the principal known-class mIoU
deliberately excludes it.

## Cross-entropy loss

For a valid pixel with target `y[p]`, categorical cross-entropy is:

```text
pixel_loss[p] = -log(p[p, y[p]])
              = -z[p, y[p]] + log(sum(exp(z[p,c]), c = 0..40))
```

The implementation passes raw logits to PyTorch so log-softmax is evaluated
stably. It does not apply softmax first. A prediction assigning 0.9 probability
to the correct class has loss 0.105; probabilities 0.5, 0.1, and 0.01 produce
losses 0.693, 2.303, and 4.605. Confident mistakes are therefore expensive.

A uniform 41-class predictor has cross-entropy:

```text
log(41) ~= 3.714
```

For unweighted evaluation, `exp(-mean_cross_entropy_loss)` is the geometric
mean probability assigned to the correct class. It is an aggregate, not the
probability of every pixel.

The gradient for class `c` is:

```text
d(pixel_loss[p]) / d(z[p,c]) = p[p,c] - 1[c == y[p]]
```

It pushes the correct logit upward and all incorrect logits downward. This
smooth, decomposable gradient is why cross-entropy is optimized instead of hard
IoU. IoU depends on argmax decisions and is not directly differentiable.

With `class_weighting: none`, every valid pixel has equal weight, not every
class. Large wall and floor regions can dominate. The moderately balanced
recipe computes an inverse-square-root weight from training-only pixel counts,
normalizes supported-class weights to mean 1, and caps them at 5. Weighted
training loss is then a weighted mean over pixels. Development and test
cross-entropy remain unweighted so experiments are comparable.

The step loss is the mean over valid pixels in the current batch. The recorded
epoch training loss is the mean of its batch means. Evaluation instead sums
cross-entropy over the complete manifest and divides by the exact valid-pixel
count. Unequal final batches or unequal ignored fractions can make those two
aggregation rules differ slightly. Training mode and evaluation mode can also
differ because stochastic model components are disabled for evaluation.

## Confusion matrix

Hard segmentation metrics derive from a 41 by 41 matrix:

```text
C[i,j] = number of pixels where y[p] == i and predicted_label[p] == j
```

Rows are ground truth and columns are predictions. For class `c`:

```text
TP[c] = C[c,c]
FN[c] = sum(C[c,j], j = 0..40) - TP[c]
FP[c] = sum(C[i,c], i = 0..40) - TP[c]
```

The row-normalized matrix answers, "Given the true class, where did its pixels
go?" Off-diagonal cells reveal systematic confusions such as chair to sofa.

## Hard-label segmentation metrics

| Metric | Computation | Intuition and use |
|---|---|---|
| Per-class IoU | `TP[c] / (TP[c] + FP[c] + FN[c])` | Region overlap while penalizing both extra and missed pixels. This is the most important per-class measure. |
| Precision | `TP[c] / (TP[c] + FP[c])` | Of pixels predicted as class `c`, how many were correct? Low precision indicates overprediction. |
| Recall | `TP[c] / (TP[c] + FN[c])` | Of true class-`c` pixels, how many were recovered? Low recall indicates missed regions. |
| F1/Dice | `2 TP[c] / (2 TP[c] + FP[c] + FN[c])` | Harmonic balance of precision and recall. For the same mask, `F1 = 2 IoU / (1 + IoU)`. |
| Overall pixel accuracy | `sum(TP[c]) / sum(C[i,j])` | Fraction of all valid pixels correct. Useful but easily dominated by large surfaces. |
| Mean class recall | Mean recall over ground-truth-present classes 0--40 | Gives every present class equal weight; includes `unknown` if present. |
| Frequency-weighted IoU | `sum((support[c] / N) * IoU[c])` | Weights classes by their pixel prevalence. More representative of an average pixel, but can hide rare-class failure. |
| Known-class mIoU | Mean IoU over ground-truth-present classes 1--40 | Principal model-selection and reporting metric. Every included known class has equal weight; `unknown` is excluded. |
| `miou_41` | Mean IoU over ground-truth-present classes 0--40 | Includes `unknown`. Despite the name, absent classes are excluded rather than filled with zero. |
| ObjectNav-six mIoU | Mean IoU over present chair, sofa/couch, plant, bed, toilet, and TV classes | Task-specific view of the six ObjectNav goal categories. |

An undefined denominator is stored as `null`, not silently converted to zero.
The macro sets use ground-truth support greater than zero. Consequently, a
class absent from ground truth is excluded from mIoU even if false-positive
predictions give its standalone IoU a value of zero.

High pixel accuracy and modest mIoU are compatible. If walls and floors occupy
most pixels, getting them right can yield 96% accuracy while several small
classes with IoU near zero pull known-class mIoU below 50%. Report both, but use
known-class mIoU as the principal quality measure.

## Global and scene-macro aggregation

Global metrics first add every evaluated pixel to one confusion matrix and then
compute IoU. Larger scenes and more prevalent classes contribute more counts.

Scene-macro evaluation instead computes known-class mIoU separately for every
scene and reports the mean and median of those scene values. It also resamples
scenes with replacement to obtain a seeded percentile-bootstrap 95% interval.
This interval describes variation across scenes; it is not a per-pixel
confidence interval or a Bayesian posterior.

Both views matter. Global mIoU measures aggregate pixel performance; scene-macro
mIoU reveals whether performance is stable across environments.

## Probability-quality metrics

| Metric | Computation | Interpretation |
|---|---|---|
| Negative log-likelihood | `-mean(log(p[p, y[p]]))` | Same mathematical quantity as unweighted evaluation cross-entropy at temperature 1. Lower is better and confident errors are penalized strongly. |
| Multiclass Brier | `mean(sum((p[p,c] - 1[c == y[p]])^2))` | Squared probability error. Perfect is 0; a confidently wrong categorical prediction approaches 2. |
| ECE | `sum((count[b] / N) * abs(accuracy[b] - confidence[b]))` | Fifteen-bin approximation of whether stated confidence matches empirical correctness. Lower is better and the value depends on binning. |
| Entropy | `-sum(p[p,c] * log(p[p,c]))` | Per-pixel uncertainty. Zero is fully concentrated; a uniform 41-way prediction has entropy `log(41)`. Correct predictions should generally have lower entropy than errors. |
| Risk-coverage | `coverage = retained/N`; `risk = 1 - retained accuracy` | Tests selective prediction: discarding low-confidence pixels should reduce risk. |

Temperature calibration must be fit on dedicated calibration-fit scenes and
reported on disjoint calibration-evaluation scenes. It can improve NLL, Brier,
ECE, and risk-coverage while leaving IoU, accuracy, and all argmax-derived
metrics unchanged.

## Optimization diagnostics

These diagnose training mechanics but are not model-quality metrics:

- learning rate verifies warm-up and cosine decay for both parameter groups;
- gradient norm is the total norm returned before clipping and exposes spikes,
  vanishing updates, or non-finite behavior;
- samples per second and seconds per optimizer step reveal stalls or data-loader
  bottlenecks;
- peak GPU memory checks headroom and supports reproducible efficiency reports;
- parameter counts verify whether the intended portion of the model is trainable.

Do not select a scientific model because it has higher throughput or smoother
gradients. Use those values to establish that optimization ran as intended.

## Artifact map after `train`

Every run created by the current code contains the following paths relative to
the run directory printed by `hm3d-semseg train`:

| Path | Contents |
|---|---|
| `metrics.jsonl` | Authoritative append-only step, epoch, development, and early-stopping records. Step records contain loss, gradient norm, both learning rates, processed samples, timing, throughput, and peak GPU memory. Stored epoch numbers are zero-based. |
| `metrics_summary.json` | Compact initial/final/minimum training loss, epoch-loss history, gradient and efficiency summary, development history and best development mIoU, links to plots, and optional tiny-subset diagnostic. |
| `summary.json` | Run-level result: sample/scene counts, device, step plan, parameter groups, checkpoint-selection value, and paths to `metrics_summary.json` and metric plots. |
| `plots/loss_and_learning_rate.png` | Step cross-entropy and both learning-rate schedules. |
| `plots/optimization_diagnostics.png` | Gradient norm, throughput, step duration, and peak GPU memory when CUDA is used. |
| `tensorboard/events.out.tfevents.*` | Interactive step/epoch scalars with smoothing, exact values, and wall time. |
| `parameter_counts.json` | Trainable/total scalar parameters and optimizer-group sizes, base rates, and weight decay. |
| `checkpoints/{best,last}/checkpoint.json` | Epoch, optimizer step, selection metric, model identity, camera/resize contract, and early-stopping state. |
| `diagnostics/training_progress/qualitative/epoch_EEE.png` | Fixed RGB, target overlay, and prediction overlay for visual training progression. |

When a development dataset is configured, the run additionally contains:

| Path | Contents |
|---|---|
| `plots/development_metrics.png` | Training/development cross-entropy and development known-class mIoU by epoch. |
| `evaluation-epoch-EEE/summary.json` | Complete development loss, global/per-class/ObjectNav-six metrics, scene-macro statistics, probability metrics, and confusion matrices for that epoch. |
| `evaluation-epoch-EEE/confusion_matrix.npy` | Exact integer confusion counts for independent analysis. |
| `evaluation-epoch-EEE/plots/*.png` | Raw and row-normalized confusion, per-class IoU, per-scene mIoU, reliability, and risk-coverage plots. |

`metrics_summary.json` identifies the `evaluation-epoch-EEE` report and plot
directory corresponding to the best development known-class mIoU.

When `training.evaluate_train_subset: true`, as in tiny overfit, the run also
contains:

| Path | Contents |
|---|---|
| `diagnostics/train_subset/summary.json` | Exact unaugmented selected-subset loss, pixel accuracy, known-class mIoU, all global/per-class metrics, and per-sample results. |
| `diagnostics/train_subset/confusion_matrix.npy` | Exact selected-subset confusion counts. |
| `diagnostics/train_subset/plots/per_sample_pixel_accuracy.png` | Memorization accuracy for each selected image. |
| `diagnostics/train_subset/plots/per_class_iou.png` | IoU for supported classes in the selected images. |
| `diagnostics/train_subset/plots/confusion_row_normalized.png` | Selected-subset error transitions by true class. |
| `diagnostics/train_subset/qualitative/*.png` | RGB, target, and prediction panels for every selected sample. |

## Artifacts after explicit evaluation

Held-out testing is deliberately not run automatically after training. Running
it repeatedly during recipe development would turn the held-out set into
another development set. Once the checkpoint and protocol are frozen, run
`hm3d-semseg evaluate` with an explicit output directory. It writes:

- `summary.json`: complete loss, hard-label, scene-macro, and probability metrics;
- `confusion_matrix.npy`: exact raw counts;
- `plots/confusion_raw.png` and `confusion_row_normalized.png`;
- `plots/per_class_iou.png` and `per_scene_miou.png`;
- `plots/reliability.png` and `risk_coverage.png`.

Calibration writes `calibration.json` inside the calibrated checkpoint with the
temperature, fit scenes, optimizer settings, and initial/final fit NLL. Evaluate
that checkpoint on disjoint calibration-evaluation scenes to obtain comparable
post-calibration probability metrics and plots.

For a run without a development dataset, `checkpoints/best` means minimum
recorded training epoch loss, not best generalization. Tiny overfit uses it for
the memorization diagnostic. A final all-training-scenes recipe normally reports
the deliberately chosen `last` checkpoint on the untouched evaluation split.

The implementations are
[training loss](../src/hm3d_semseg/models/segformer.py),
[training reporting](../src/hm3d_semseg/training/reporting.py),
[confusion metrics](../src/hm3d_semseg/evaluation/metrics.py), and
[calibration metrics](../src/hm3d_semseg/calibration/metrics.py).
