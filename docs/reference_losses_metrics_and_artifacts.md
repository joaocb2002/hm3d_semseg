# Reference: losses, metrics, model selection, and artifacts

This reference answers four separate questions:

1. **Optimization:** what scalar gives the model a gradient?
2. **Model selection:** which checkpoint generalizes best during development?
3. **Final evaluation:** how good are the frozen segmentation regions?
4. **Calibration:** do the reported probabilities match empirical correctness?

Do not use one quantity as a substitute for another. Training cross-entropy
drives weight updates; development known-class mIoU selects a candidate
checkpoint; official-validation metrics report the frozen model; calibration
metrics assess confidence reliability.

Equations use standard Markdown math delimiters: `$...$` inline and `$$...$$`
for display equations. In VS Code use **Open Preview** and keep
`markdown.math.enabled` enabled; formulas are no longer represented as
shell/text code blocks.

## What matters at each stage

| Stage | Primary evidence | Supporting evidence | Not established |
|---|---|---|---|
| Tiny overfit | Falling training cross-entropy and near-memorization of four selected images | Selected-subset accuracy/mIoU, confusion matrix, qualitative alignment | Generalization |
| Recipe development | Best **development known-class mIoU** | Per-class/ObjectNav-six IoU, scene-macro statistics, development loss, confusion and qualitative panels | Unbiased official performance after repeated choices |
| Final fixed evaluation | **Official known-class mIoU** | Per-class IoU, ObjectNav-six, scene-macro interval, pixel accuracy, unknown behavior | Reliable probabilities unless calibration is evaluated |
| Calibrated evaluation | **NLL, multiclass Brier, ECE, reliability, risk-coverage** on disjoint calibration-evaluation scenes | Hard-label metrics, which must remain unchanged | Improved segmentation regions; positive temperature cannot change argmax |

Mean average precision (mAP) is not used. It is mainly an object-detection or
instance-segmentation metric based on confidence-ranked instance matches. This
project predicts one semantic class per pixel and does not identify instances.

## Symbols used throughout

| Symbol | Meaning |
|---|---|
| $u$ | One spatial pixel location; never a probability |
| $c$ | A model class index in $\{0,\ldots,40\}$ |
| $z_{u,c}$ | Model logit for class $c$ at location $u$ |
| $q_{u,c}(T)$ | Softmax probability for class $c$ at location $u$, using temperature $T>0$ |
| $y_u$ | Ground-truth target at location $u$ |
| $\hat{y}_u$ | Predicted hard class at location $u$ |
| $\mathcal V$ | Valid evaluated locations: all locations whose target is not 255 |
| $\mathbf{1}[A]$ | 1 when statement $A$ is true, otherwise 0 |

Model class 0 is learnable `unknown`. Classes 1--40 are the MPCAT40 classes.
Target 255 is `ignore_index`: it has no logit and belongs to neither loss nor
metrics.

## From logits to probabilities and labels

SegFormer produces 41 logits per location. They are bilinearly upsampled to the
native mask dimensions. Softmax with temperature $T$ is

$$
q_{u,c}(T)
=
\frac{\exp(z_{u,c}/T)}
     {\sum_{k=0}^{40}\exp(z_{u,k}/T)}.
$$

The hard label and confidence are

$$
\hat{y}_u = \underset{c\in\{0,\ldots,40\}}{\operatorname{argmax}}\;q_{u,c}(T),
\qquad
\operatorname{confidence}_u = \max_c q_{u,c}(T).
$$

Softmax and division by a positive temperature preserve logit ordering.
Therefore

$$
\operatorname{argmax}_c z_{u,c}
=
\operatorname{argmax}_c q_{u,c}(T)
$$

for every positive $T$. Calibration changes confidence, not the segmentation
mask.

## Training objective: categorical cross-entropy

At training temperature $T=1$, the loss for one valid location is

$$
\ell_u
=
-\log q_{u,y_u}(1)
=
-z_{u,y_u}+\log\left(\sum_{c=0}^{40}\exp z_{u,c}\right).
$$

The unweighted mean over valid locations is

$$
\mathcal L_{\mathrm{CE}}
=
\frac{1}{|\mathcal V|}
\sum_{u\in\mathcal V}\ell_u.
$$

The implementation passes raw logits to PyTorch cross-entropy. PyTorch combines
log-softmax and negative log-likelihood stably; the code does not calculate a
softmax tensor first.

Intuitively, cross-entropy asks: **how much probability did the model assign to
the true class?** Correct-class probabilities 0.9, 0.5, 0.1, and 0.01 yield
losses approximately 0.105, 0.693, 2.303, and 4.605. Confident errors are
penalized strongly. A uniform 41-class predictor has loss

$$
\log 41 \approx 3.714.
$$

The per-logit gradient is

$$
\frac{\partial\ell_u}{\partial z_{u,c}}
=
q_{u,c}(1)-\mathbf{1}[c=y_u].
$$

It raises the correct-class logit and lowers incorrect logits. This smooth,
location-decomposable gradient is why cross-entropy drives training. Hard IoU
depends on argmax decisions and is not directly differentiable.

Loss does not by itself decide when the recommended full experiment stops. It
runs until `max_optimizer_steps` (or its epoch safety cap), while the best
checkpoint is selected by development known-class mIoU.
When early stopping is enabled, improvement is judged by development
known-class mIoU if a development root exists; otherwise it uses negative
training epoch loss.

### Historical moderately balanced weighted loss

Let $n_c$ be the valid training-pixel count for supported class $c$, and let
$S$ be the set of classes with $n_c>0$. Before mean normalization and
capping, the inverse-square-root value is

$$
\widetilde w_c
=
\sqrt{\frac{\sum_{k\in S} n_k}{|S|\,n_c}}.
$$

Supported weights are normalized to mean 1 and capped at 5. Unsupported
training classes receive zero weight. The batch objective is equivalent to

$$
\mathcal L_{\mathrm{weighted}}
=
\frac{\sum_{u\in\mathcal V} w_{y_u}\,\ell_u}
     {\sum_{u\in\mathcal V} w_{y_u}}.
$$

This moderates dominance by large surfaces without giving extremely rare
classes unbounded influence. It is not a universal segmentation standard. In
this project's completed controlled comparison, it did not materially improve
held-out known-class mIoU and worsened several important supporting measures.
The recommended ADE20K-style recipe therefore uses unweighted cross-entropy.
The implementation and historical configs remain for reproducibility. Their
exact vector and SHA-256 are in `class_weights.npy` and `class_weights.json`.

Lovasz-Softmax, Dice, focal loss, and mixtures of them are legitimate research
options, but none is guaranteed to improve this dataset. Adding one while also
changing augmentation, schedule, batch, and optimizer grouping would prevent a
causal interpretation. Test such a loss only as a later single-variable
ablation if the corrected cross-entropy recipe leaves a specific failure.

### How reported training and evaluation loss differ

- Step loss is PyTorch's weighted or unweighted mean over valid locations in
  the current optimization batch.
- Recorded training epoch loss is the mean of recorded batch means.
- Evaluation loss sums unweighted negative log-likelihood over all valid
  locations and divides once by the exact valid-location count.

Unequal final batches or unequal ignored fractions can cause small aggregation
differences. Evaluation also disables stochastic training behavior.

## Confusion matrix: source of hard-label metrics

The 41×41 global confusion matrix is

$$
C_{i,j}
=
\sum_{u\in\mathcal V}
\mathbf{1}[y_u=i]\,\mathbf{1}[\hat y_u=j].
$$

Rows are ground truth and columns are predictions. For class $c$,

$$
\mathrm{TP}_c=C_{c,c},
$$

$$
\mathrm{FN}_c=\sum_j C_{c,j}-\mathrm{TP}_c,
\qquad
\mathrm{FP}_c=\sum_i C_{i,c}-\mathrm{TP}_c.
$$

The row-normalized matrix answers: “given that the true class was $i$, where
did those locations go?” Off-diagonal cells reveal systematic confusion.

## Hard-label segmentation metrics

For a class $c$:

$$
\mathrm{IoU}_c
=
\frac{\mathrm{TP}_c}
     {\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c},
$$

$$
\mathrm{Precision}_c
=
\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FP}_c},
\qquad
\mathrm{Recall}_c
=
\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FN}_c},
$$

$$
\mathrm{F1}_c
=
\frac{2\mathrm{TP}_c}
     {2\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c}.
$$

- **IoU** measures region overlap and penalizes both extra and missed area. It
  is the principal per-class segmentation measure.
- **Precision** asks how many predicted class-$c$ locations were correct. Low
  precision means overprediction.
- **Recall** asks how many true class-$c$ locations were recovered. Low recall
  means missed regions.
- **F1/Dice** balances precision and recall. For the same binary mask,
  (\mathrm{F1}=2\mathrm{IoU}/(1+\mathrm{IoU})).

Let $s_c=\sum_j C_{c,j}$ be ground-truth support. The principal macro set is

$$
\mathcal K=\{c\in\{1,\ldots,40\}:s_c>0\}.
$$

Known-class mIoU is

$$
\mathrm{mIoU}_{\mathrm{known}}
=
\frac{1}{|\mathcal K|}\sum_{c\in\mathcal K}\mathrm{IoU}_c.
$$

It gives every present known class equal weight and excludes `unknown`. This is
the checkpoint-selection and primary reporting metric.

Other summaries are:

- **`miou_41`:** mean IoU over ground-truth-present classes 0--40, including
  `unknown` when present. Despite its name, absent classes are excluded.
- **ObjectNav-six mIoU:** mean over present chair, sofa/couch, plant, bed,
  toilet, and TV classes.
- **Overall pixel accuracy:**
  (\sum_c\mathrm{TP}_c / \sum_{i,j}C_{i,j}). It can be dominated by walls,
  floors, and ceilings.
- **Mean class recall:** mean recall over ground-truth-present classes 0--40.
- **Frequency-weighted IoU:**
  $\sum_c (s_c/N)\mathrm{IoU}_c$, emphasizing the average location rather
  than the average class.

An undefined denominator is stored as `null`, not zero. A class absent from
ground truth is excluded from macro means even if false-positive predictions
make its standalone IoU zero. High pixel accuracy and modest mIoU can therefore
coexist; report both, but select by known-class mIoU.

## Global versus scene-macro aggregation

**Global metrics** first accumulate every evaluated location into one confusion
matrix. Large scenes and prevalent classes contribute more counts.

In an ordinary smoke or full recipe-development run, every accuracy, IoU,
per-class, per-scene, and probability metric is a **development-set** metric.
The only routine training-set curve is the optimization cross-entropy. The
pipeline does not add a costly second evaluation over the complete training
set; tiny overfit is the explicit exception and labels its four-image result as
a memorization diagnostic.

**Scene-macro metrics** compute known-class mIoU independently per scene, then
report the mean and median. They reveal whether performance is stable across
environments rather than concentrated in large/easy scenes.

The scene-bootstrap interval resamples scene scores with replacement. With 15
development scenes and `bootstrap_samples: 1000`, each bootstrap draw samples
15 scene scores, computes their mean, and the evaluator reports the 2.5th and
97.5th percentiles of 1,000 means. This interval describes between-scene
variation; it is neither a per-location confidence interval nor a Bayesian
posterior. The fixed seed makes it reproducible. More resamples smooth the
numerical percentile estimate but do not create more evaluation information.

## Probability-quality metrics

All formulas below use only valid locations and probabilities after the stated
temperature.

### Negative log-likelihood

$$
\mathrm{NLL}
=
-\frac{1}{|\mathcal V|}
\sum_{u\in\mathcal V}\log q_{u,y_u}(T).
$$

At $T=1$, this is the same mathematical quantity as unweighted evaluation
cross-entropy. Lower is better; confident errors are costly.

### Multiclass Brier score

$$
\mathrm{Brier}
=
\frac{1}{|\mathcal V|}
\sum_{u\in\mathcal V}
\sum_{c=0}^{40}
\left(q_{u,c}(T)-\mathbf{1}[c=y_u]\right)^2.
$$

It is a squared error over the complete probability vector. Perfect is 0; a
confidently wrong categorical prediction approaches 2.

### Expected calibration error

Partition locations by confidence into bins $B_b$. For each nonempty bin,
let `acc` be empirical accuracy and `conf` be mean stated confidence:

$$
\mathrm{ECE}
=
\sum_b
\frac{|B_b|}{|\mathcal V|}
\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.
$$

Lower is better. ECE is a bin-dependent approximation: 15 equal-width bins is
the project default. `calibration_bins` affects ECE, reliability plots, and
risk-coverage grouping; it does not limit images or participate in temperature
optimization.

### Entropy and selective risk

Per-location entropy is

$$
H_u=-\sum_{c=0}^{40}q_{u,c}(T)\log q_{u,c}(T).
$$

Zero means all mass is on one class; uniform uncertainty is $\log 41$.
Incorrect predictions should generally have higher entropy than correct ones.

Risk-coverage sorts or bins by confidence. Coverage is the retained fraction;
risk is one minus retained accuracy. A useful uncertainty estimate lets the
system discard low-confidence locations and reduce risk. Every point in
`risk_coverage.png` is labeled with its minimum retained softmax-confidence
threshold $t$; the plot means "keep locations whose confidence is at least
$t$." The final point at coverage 1 therefore has risk equal to one minus
overall pixel accuracy.

## What temperature calibration optimizes

Calibration freezes every SegFormer weight and parameterizes

$$
T=\exp(\tau),
$$

with a positive, clamped temperature. Adam updates only scalar $\tau$ to
minimize NLL on `calibration-fit-v1`. A value $T>1$ softens an overconfident
distribution; $0<T<1$ sharpens an underconfident one. The chosen $T$ is then
evaluated on disjoint `calibration-evaluation-v1` scenes.

The intended interpretation is: among many locations assigned approximately
confidence $r$, approximately fraction $r$ should be correct. One global
temperature can correct global sharpness; it cannot guarantee perfect
class-conditional, scene-conditional, or individual-location calibration.

Calibration may improve NLL, Brier, ECE, reliability, and risk-coverage. It
cannot improve IoU, accuracy, or any other argmax-derived metric.

## Optimization diagnostics are not quality metrics

- learning rates verify warm-up and polynomial decay for decay/no-decay
  pretrained and decode-head groups;
- gradient norm exposes spikes, vanishing updates, or non-finite behavior;
- samples per second and seconds per optimizer step reveal loader/GPU stalls;
- peak GPU memory checks headroom;
- parameter counts verify which parts of the network are trainable.

Use these to diagnose mechanics and efficiency, not to select the scientific
model.

## JSON versus JSONL

- **JSON** stores one complete structured value. Files such as
  `records/run_summary.json` and `provenance/provenance.json` are rewritten
  atomically as complete documents.
- **JSONL** stores one independent JSON object per line.
  `records/metrics.jsonl` is
  append-only, so step/epoch records can stream during training and survive an
  interruption without rebuilding one huge JSON array.

`records/metrics.jsonl` is the authoritative time series.
`records/metrics_summary.json` is a compact derived summary of that series.

## Human report versus raw record

Every completed training run automatically creates the run-root `index.html`. Start
there for interpretation, then use the linked CSV/JSON sources when exact
values or scripted analysis are needed. The report can be regenerated without
loading a model:

```bash
hm3d-semseg report-run --run /absolute/path/to/run
```

Current workstation example:

```bash
hm3d-semseg report-run \
  --run /home/joaocb2002/hm3d-semseg-data/runs/segformer_b2_baseline_smoke
```

Regeneration refreshes the run-root `index.html`, `report/`,
`records/report_data.json`, and `provenance/artifact_manifest.json`. It does not
alter `records/metrics.jsonl`, per-epoch evaluation JSON, checkpoints, run
provenance, or TensorBoard events. Thus machine-readable truth and human
presentation stay separate.

For recipe selection, compare two complete development runs directly:

```bash
hm3d-semseg compare-runs \
  --run /absolute/path/to/baseline \
  --run /absolute/path/to/balanced \
  --output /absolute/path/to/comparison
```

The comparison checks protocol-identifying provenance, overlays held-out
curves, tabulates compute, reports per-class changes, computes paired
scene-by-scene mIoU differences with a bootstrap interval, and places the fixed
best-epoch development contact sheets side by side when available. It never
ranks differently weighted recipes by their training-loss magnitudes.

## Artifact map after `train`

Every run directory printed by `hm3d-semseg train` uses one stable hierarchy:

```text
RUN/
├── index.html                         # start here
├── checkpoints/{best,last}/
├── checkpoints/min_development_loss/           # optional configured diagnostic
├── tensorboard/
├── records/                           # authoritative machine-readable records
│   ├── metrics.jsonl
│   ├── metrics_summary.json
│   ├── run_summary.json
│   └── report_data.json
├── provenance/                        # identity and reproducibility evidence
│   ├── resolved_config.yaml
│   ├── provenance.json
│   ├── parameter_counts.json
│   ├── class_weights.{json,npy}       # only when generated by the recipe
│   └── artifact_manifest.json
├── report/                            # compact human-readable derivatives
│   ├── summary.md
│   ├── tables/
│   └── summary_metrics_plots/
│       ├── overview/
│       ├── segmentation/
│       ├── classes_and_scenes/
│       ├── probability/
│       └── optimization/
└── diagnostics/
    ├── qualitative/{train,development}/
    ├── epoch_evaluations/development/epoch_EEE/
    └── train_subset/                  # tiny-overfit only
```

The separation is intentional: `records/` is for parsers, `provenance/` proves
what ran, `report/` is for immediate human interpretation, and `diagnostics/`
contains detailed qualitative and per-epoch evidence. Nothing in the report
replaces or deletes the authoritative record.

| Relative path | Contents |
|---|---|
| `index.html` | Primary static dashboard with warnings, scoped headline metrics, checkpoint/epoch/class tables, organized plots, and fixed-view epoch sliders. |
| `records/metrics.jsonl` | Append-only step, epoch, development, and early-stopping records. Epoch numbers are zero-based. |
| `records/metrics_summary.json` | Initial/final/minimum training loss, histories, optimization summary, development best/history, and plot links. |
| `records/run_summary.json` | Run-level counts, device, step plan, parameter groups, checkpoint-selection value, and major artifact paths. |
| `records/report_data.json` | Complete structured data consumed by the human report. |
| `provenance/resolved_config.yaml` | Exact merged scientific and host configuration used by the process. |
| `provenance/provenance.json` | Git, packages, hardware, model revision, datasets, selected sample IDs, and reproducibility settings. |
| `provenance/parameter_counts.json` | Trainable/total parameters and optimizer-group sizes/base rates/weight decay. |
| `provenance/artifact_manifest.json` | Source and generated-file hashes for report reproducibility. |
| `report/summary_metrics_plots/overview/` | Training/development loss plus the main global, scene-macro, and accuracy/recall curves. |
| `report/summary_metrics_plots/segmentation/` | Full-scale and zoomed segmentation curves plus ObjectNav-six class histories. |
| `report/summary_metrics_plots/classes_and_scenes/` | Class/scene heatmaps, selected-checkpoint precision/recall/IoU, confusions, scene distribution, and checkpoint changes. |
| `report/summary_metrics_plots/probability/` | NLL, Brier, ECE, entropy, reliability, and confidence-threshold-labelled risk/coverage. These do not replace hard-segmentation metrics. |
| `report/summary_metrics_plots/optimization/` | Aggregated training loss, learning rates, finite-gradient/AMP health, throughput, step duration, and CUDA memory. Raw steps remain in JSONL. |
| `report/summary.md` | Compact text report suitable for review or an experiment note. |
| `report/tables/*.csv` | Epoch, checkpoint, best-class, ObjectNav-six, best-scene, and largest-confusion tables. |
| `tensorboard/events.out.tfevents.*` | Interactive scalars plus fixed train/development contact sheets. |
| `checkpoints/{best,last}/checkpoint.json` | Epoch, step, selection value, model/camera/resize identity, and stopping state. |
| `checkpoints/{best,last}/model.safetensors` | Learned SegFormer tensor values for that checkpoint. |
| `checkpoints/min_development_loss/` | Optional complete checkpoint at minimum development cross-entropy when explicitly enabled. |
| `diagnostics/qualitative/selection.json` | Prediction-independent fixed sample IDs, scenes, ground-truth class coverage, policy, and seed. |
| `diagnostics/qualitative/{train,development}/contact_sheets/epoch_EEE.png` | Up to ten fixed rows: RGB, ground truth, prediction, correct/error map, and confidence. |
| `diagnostics/qualitative/{train,development}/samples/` | RGB/ground truth stored once; per-epoch prediction, error, confidence, and sample metrics. |

When a development dataset is configured, each
`diagnostics/epoch_evaluations/development/epoch_EEE/` directory contains its
complete `summary.json`, exact `confusion_matrix.npy`, qualitative evidence,
and detailed plots. `records/metrics_summary.json` points to the evaluation and
plots for the highest development known-class mIoU. That epoch supplies
`checkpoints/best`.

The qualitative views are not selected by model accuracy. Up to ten views per
active split are fixed before training from manifest ground-truth statistics,
favoring distinct scenes plus broad/rare class coverage. Development
predictions are captured inside the evaluation pass, so they add no extra
model inference. Training capture adds at most ten unaugmented forwards per
recorded epoch. With `qualitative_every_epochs: 1`, the temporal sequence is
complete; increasing the value reduces diagnostic I/O.

The contact-sheet error colors are green for correct valid pixels, red for
incorrect valid pixels, and gray for ignored target 255. Confidence is the
maximum of the 41-way softmax and is displayed in grayscale. These views are
diagnostics; exact aggregate evidence remains in the global confusion matrix.

When `evaluate_train_subset: true`, as in tiny overfit and the generalization
probe, the evaluator uses either the complete limited training set or the
deterministic scene-diverse limit in `train_subset_evaluation_samples`:

| Relative path | Contents |
|---|---|
| `diagnostics/train_subset/summary.json` | Unaugmented selected-subset loss, accuracy, known-class mIoU, per-class and per-sample results. |
| `diagnostics/train_subset/confusion_matrix.npy` | Exact selected-subset counts. |
| `diagnostics/train_subset/plots/*.png` | Per-sample accuracy, per-class IoU, and normalized confusion. |
| `diagnostics/train_subset/qualitative/*.png` | RGB, target, and prediction for every selected sample. |

These are memorization/generalization-gap diagnostics on training images, not
held-out evidence. `train_subset_evaluation_samples` never limits the images
used to update weights.

When `save_min_development_loss_checkpoint: true`, the complete checkpoint at
the lowest observed development cross-entropy is retained under
`checkpoints/min_development_loss`. The normal `checkpoints/best` remains the
highest development known-class-mIoU checkpoint and continues to control early
stopping; `checkpoints/last` remains the final executed epoch.

## Artifacts after evaluation, calibration, and benchmarking

An explicit `evaluate --output OUTPUT` writes:

- `OUTPUT/summary.json` with loss, hard-label, scene-macro, and probability
  metrics;
- `OUTPUT/confusion_matrix.npy` with exact raw counts;
- `OUTPUT/plots/confusion_raw.png` and
  `confusion_row_normalized.png`;
- `OUTPUT/plots/per_class_iou.png` and `per_scene_miou.png`;
- `OUTPUT/plots/reliability.png` and `risk_coverage.png`.
- `OUTPUT/qualitative/selection.json`, compact sample maps, and one ten-view
  contact sheet selected without using predictions;
- `OUTPUT/report/index.html`, `summary.md`, CSV metric/confusion tables, and a
  report manifest.

The explicit-evaluation qualitative set is captured inside the one evaluation
pass, so it does not run the model twice. This applies equally to official
validation and the later calibrated evaluation.

Calibration copies the full source checkpoint to `checkpoints/calibrated` and
adds `calibration.json` with temperature, fit scenes, optimizer settings, and
the first/last recorded optimization-batch NLL. Benchmarking writes its
hardware, precision, latency,
throughput, memory, and checkpoint-size report under the explicit benchmark
output.

For a run without development data, `checkpoints/best` means minimum training
epoch loss, not best generalization. Tiny overfit uses it for memorization. The
final all-training-scenes refit uses its deliberately scheduled
`checkpoints/last` for official evaluation.

Return to [server training](08_server_training.md),
[server evaluation/calibration](09_server_evaluation_and_calibration.md), or the
[execution guide](README.md).
