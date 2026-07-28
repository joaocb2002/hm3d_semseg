# 7. Training

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

Use only two to four samples. Loss must fall clearly and training predictions
must approach memorization. There is no universal numeric threshold, but failure
to achieve roughly near-perfect supported-pixel accuracy after the loss plateaus
blocks full training; inspect mask alignment, class 0, resize, loss, and learning
rate.

Development baseline:

```bash
hm3d-semseg train \
  --config configs/experiments/segformer_b2_baseline.yaml \
  --local-config configs/local.yaml
```

The complete model is fine-tuned. AdamW uses a lower pretrained-parameter LR and
higher new-classifier LR. Loss is raw-logit cross-entropy with ignore 255.
Cosine decay, clipping, AMP, accumulation, deterministic seeds, atomic best/last
checkpoints, optimizer/scheduler/scaler resume, and JSONL metrics are included.

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

Runs contain resolved config, provenance, parameter counts, metrics,
checkpoints, plots/qualitative directories, evaluations, and summary. Resume by
setting `training.resume` to a prior `checkpoints/last`; run history is appended.

After development, freeze recipe and duration. Generate all 145 train scenes,
set development dataset null, choose a new final run name, and train without
tuning on official validation.

Next: [evaluation and calibration](08_evaluation_and_calibration.md).
