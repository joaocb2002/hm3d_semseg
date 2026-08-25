# 7. Training

**Execution location: GPU server.** Enter this step with the exact committed
repository clone, validated transferred datasets, pinned model snapshot, and a
fresh server-local `configs/local.yaml`. This step writes only to the external
server run root; it does not put learned weights into Git. Remain on the server
through step 08.

Leave the rendering environment and run every command in this guide from the
host-matched training environment. When training on a dedicated GPU server,
complete the [workstation-to-server handoff](06a_server_handoff.md) first; the
server needs the repository and complete offline datasets, not Habitat or HM3D
source assets.

```bash
conda activate hm3d-semseg-train
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
```

First download the checkpoint explicitly:

```bash
hm3d-semseg download-model \
  --local-config configs/local.yaml \
  --model-id nvidia/segformer-b2-finetuned-ade-512-512
```

The command resolves remote revision to a commit and records its snapshot under
`paths.cache_root`. Put the printed `resolved_revision` in
`model.revision` inside `configs/local.yaml`. Training searches that configured
cache and uses `local_files_only: true`; it will not silently redownload.
Review the checkpoint license; weights are not committed or redistributed.

Set absolute `training.train_dataset` and optional
`training.development_dataset` in `configs/local.yaml`. Validate both first.
For the tiny-overfit diagnostic, keep `development_dataset: null`; otherwise it
would evaluate the complete development set after every diagnostic epoch.
`training.device` defaults to `auto`: before dataset scanning or model loading,
the runtime executes a real kernel on each visible GPU and selects the working
device with the most free memory. Set `cpu` or `cuda:N` only when an experiment
requires an explicit device. CUDA that is visible but architecture-incompatible
fails immediately with an installer command instead of silently falling back.

Tiny-overfit:

```bash
hm3d-semseg train \
  --config configs/experiments/overfit_tiny.yaml \
  --local-config configs/local.yaml
```

Training prints the selected device, effective sample count, batching and
optimizer-step plan, AMP state, and trainable/total parameter counts. One
optimizer-step progress bar reports completed/remaining steps, elapsed time,
ETA, current epoch, loss, learning rate, and throughput. It is enabled by
default; add `--no-progress` for quiet batch logs.

`training.max_train_samples` controls the diagnostic subset and
`training.sample_selection: scene_diverse` takes at most one view from each
scene before reusing a scene. Verify the configured count before launching; a
very small subset tests memorization, while a larger subset is a slower training
sanity check. The seeded IDs are recorded in resolved provenance and the
summary, so repeated runs use the same samples. Limited runs fully decode and
validate only their selected files while still checking the complete manifest,
schema, hashes, duplicate IDs, and scene-split contract. Runs without
`max_train_samples` retain complete file validation.

After training, `training.evaluate_train_subset: true` evaluates the best
checkpoint on every selected sample. Review
`diagnostics/train_subset/summary.json`, its per-sample accuracy and per-class
IoU plots, row-normalized confusion matrix, and labeled qualitative panels.
These are memorization measurements on training images, not generalization
results. Loss must fall clearly and predictions must approach memorization.
There is no universal numeric threshold, but failure to achieve roughly
near-perfect supported-pixel accuracy after the loss plateaus blocks full
training; inspect mask alignment, class 0, resize, loss, and learning rate.

To inspect the richer interactive log in a browser, use the actual run directory
printed by the command:

```bash
# Replace the suffix with the actual run directory printed by training.
OVERFIT_RUN_DIR=/home/joaocb2002/hm3d-semseg-data/runs/overfit_tiny-002
tensorboard --logdir "$OVERFIT_RUN_DIR/tensorboard"
```

Open `http://localhost:6006`. The saved PNG is a convenient static loss/LR
snapshot; TensorBoard additionally provides zooming, smoothing, exact values,
wall time, gradient norm, throughput, GPU memory, and diagnostic/development
metrics. On a remote server, forward port 6006 over SSH rather than exposing the
TensorBoard server publicly.

The two learning-rate curves use one cosine multiplier: a scalar schedule that
moves from 1 to 0 over the planned optimizer steps. Multiplying both base rates
by the same scalar preserves their 10:1 classifier/pretrained ratio until both
reach zero at the end.

Development baseline:

```bash
# First set this absolute path in configs/local.yaml:
# training.development_dataset: /absolute/path/to/development-v1

hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

The complete model is fine-tuned. AdamW uses a lower pretrained-parameter LR and
higher new-classifier LR. Loss is raw-logit cross-entropy with ignore 255.
Cosine decay, clipping, AMP, accumulation, deterministic seeds, atomic best/last
checkpoints, optimizer/scheduler/scaler resume, and JSONL metrics are included.
See [losses, metrics, and run artifacts](losses_and_metrics.md) for the exact
formulas, interpretation, checkpoint-selection rules, and output-file map.

The baseline is unweighted. After the training-pixel census, a controlled
moderate alternative is available:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_moderately_balanced.yaml \
  --local-config configs/local.yaml
```

It computes inverse-square-root weights from the training manifest only,
normalizes and caps them at 5, then saves the vector and SHA-256. Do not combine
this with nonzero class-aware oversampling.

Runs contain resolved config, provenance, parameter counts, raw
`metrics.jsonl`, compact `metrics_summary.json`, TensorBoard events, checkpoints,
plots, diagnostics, evaluations, and `summary.json`. Static plots
include loss/learning-rate and optimization diagnostics; development runs also
plot development loss and known-class mIoU. Per-epoch visual progression is
stored under `diagnostics/training_progress/qualitative`; final selected-subset
diagnostics remain under `diagnostics/train_subset`. A fresh run never mixes
with an existing directory: if the requested name exists, the new run is
allocated `<name>-002`, then `<name>-003`, and so on. Resume by setting
`training.resume` to a prior `checkpoints/last`; only an explicit resume reuses
the requested run directory and appends its history.

After development, freeze recipe and duration. Use a distinct validated
`train-all-v1` dataset containing all 145 training scenes; never extend the
130-scene fit dataset in place. Set the development dataset to null, choose a
new final run name, and train without tuning on official validation. The
[handoff guide](06a_server_handoff.md) gives the exact server and return
sequence.

Still on the GPU server, next: [evaluation and calibration](08_evaluation_and_calibration.md).
