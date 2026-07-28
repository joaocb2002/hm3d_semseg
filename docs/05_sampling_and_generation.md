# 5. Sampling and generation

The generator loads one scene once, asks the navmesh for navigable points,
rejects positions closer than the configured XZ distance, distributes yaw
headings, cycles an explicit pitch list, renders aligned sensors, rejects nearly
invalid/unknown frames, and closes the simulator. Seeds combine the global seed
and scene ID, so scene ordering does not change poses.

The validation sampler is spatial and never class-targeted. The initial
implementation also leaves training `class_aware_fraction` at zero; implement
supplementary visibility-aware sampling only after audits justify it.

Estimate and validate without rendering:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/pilot.yaml \
  --local-config configs/local.yaml \
  --dry-run
```

Generate:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/pilot.yaml \
  --local-config configs/local.yaml
```

The command prints the upper sample count and uncompressed storage estimate.
PNG is used for correctness pilots; JPEG 95 or lossless WebP may be recorded for
full RGB. Masks are always one-channel lossless PNG. Depth is float32 NPY.

Resume by rerunning exactly the same command. Existing manifest IDs are skipped.
Changing seed, split, camera, taxonomy, codec, or depth contract refuses to
resume. Failures before a manifest append may leave unreferenced atomic files;
rerunning safely replaces only the same deterministic sample.

After the complete train audit, verify the checked-in 130/15 protocol or
deliberately regenerate it:

```bash
hm3d-semseg make-dev-split \
  --local-config configs/local.yaml \
  --audit /absolute/generated/root/audits/train \
  --output configs/data/splits
```

Review the coverage report and commit any changed lists as a protocol change.
Full generation requires explicit pitches:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/train.yaml \
  --local-config configs/local.yaml \
  --split-list configs/data/splits/fit.txt

hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --split-list configs/data/splits/development.txt
```

EGL/driver failures are actionable blockers, not silently skipped frames.

Next: [dataset format](06_dataset_format.md).
