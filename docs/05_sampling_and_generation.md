# 5. Sampling and generation

**Execution location: workstation.** This is the only machine that needs the
licensed HM3D source assets and Habitat renderer. Its output is a collection of
self-contained offline dataset roots; those roots, not the simulator install,
cross to the GPU server.

Run every command in this guide from the rendering environment:

```bash
conda activate habitat
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
```

The generator loads one scene once, asks the navmesh for navigable points,
rejects positions closer than the configured XZ distance, distributes yaw
headings, cycles an explicit pitch list, renders aligned sensors, rejects nearly
invalid/unknown frames, and closes the simulator. Seeds combine the global seed
and scene ID, so scene ordering does not change poses.

At every accepted position, `yaws_per_position: 4` produces a 90-degree heading
grid. The grid at each subsequent position is advanced by
`yaw_offset_per_position_degrees: 30.0`, matching the default ObjectNav rotate
increment while avoiding the same four headings at every sampled position.

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

Validate the pilot before scaling up:

```bash
hm3d-semseg validate-dataset \
  --dataset /home/joaocb2002/hm3d-semseg-data/generated/pilot
```

Continue only when the report has `valid: true`, no errors, the expected sample
and scene counts, and manually sensible RGB/mask/overlay panels under
`pilot/validation/`.

## Complete split audits

The pilot validation does not replace the descriptor audits for the complete
official train and validation splits. Still in the `habitat` environment, run:

```bash
hm3d-semseg audit-taxonomy \
  --local-config configs/local.yaml \
  --split train \
  --output /home/joaocb2002/hm3d-semseg-data/generated/audits/train

hm3d-semseg audit-taxonomy \
  --local-config configs/local.yaml \
  --split val \
  --output /home/joaocb2002/hm3d-semseg-data/generated/audits/val
```

Accept the audits only when train reports 145 discovered/complete scenes,
validation reports 36 discovered/complete scenes, both mapping hashes equal the
frozen mapping hash, `zero_support_classes` is empty, and every ObjectNav-six
count is positive. `rendered_image_count: 0` is expected because these commands
audit semantic descriptors rather than rendered frames.

## Verify the 130/15 development protocol

Generate a candidate split in a temporary directory first. This avoids
overwriting the checked-in protocol merely to verify it:

```bash
SPLIT_CHECK_DIR="$(mktemp -d /tmp/hm3d-semseg-split-check.XXXXXX)"

hm3d-semseg make-dev-split \
  --local-config configs/local.yaml \
  --audit /home/joaocb2002/hm3d-semseg-data/generated/audits/train \
  --output "$SPLIT_CHECK_DIR"

diff -u \
  <(sed '/^[[:space:]]*$/d' configs/data/splits/fit.txt) \
  <(sed '/^[[:space:]]*$/d' "$SPLIT_CHECK_DIR/fit.txt")

diff -u \
  <(sed '/^[[:space:]]*$/d' configs/data/splits/development.txt) \
  <(sed '/^[[:space:]]*$/d' "$SPLIT_CHECK_DIR/development.txt")
```

No output from either `diff` means scene membership is identical. Confirm
`fit_scenes: 130`, `development_scenes: 15`, and no zero-support class in
`split_report.json`. Only write directly to `configs/data/splits` when
deliberately changing the protocol; review and commit such a change.

## Freeze and inspect the pitch protocol

The resolved ObjectNav configuration confirms a 30-degree look increment but
does not define repeat bounds. Record an explicit, bounded research choice in
`configs/local.yaml` before full generation. The conservative one-action choice
is:

```yaml
camera:
  profile: /home/joaocb2002/hm3d-semseg-data/generated/camera/objectnav_resolved.yaml
  pitch_degrees: [0.0, 30.0, -30.0]
  require_explicit_pitch: true
  allow_mismatch: false
```

This means neutral plus one ObjectNav look increment in either direction; it
does not infer that repeated look actions are bounded at 30 degrees. Inspect a
fresh output directory after making the edit:

```bash
hm3d-semseg inspect-scene \
  --local-config configs/local.yaml \
  --split minival \
  --scene-id 00800-TEEsavR23oF \
  --num-views 6 \
  --output /home/joaocb2002/hm3d-semseg-data/generated/inspection/TEEsavR23oF-pitch-check

jq '[.views[].pose.pitch_degrees]' \
  /home/joaocb2002/hm3d-semseg-data/generated/inspection/TEEsavR23oF-pitch-check/report.json
```

Confirm that the report cycles through `0`, `30`, and `-30`, then inspect the
RGB, mask, and overlay files for sensible views and alignment.

## Dataset lifecycle: why six roots exist

The production data is deliberately stored as six self-contained dataset
roots. They are created in two workstation stages, but each has exactly one
role later in the server protocol. They are not six interchangeable training
sets, and the development or official-validation images must never be mixed
into recipe fitting by randomly splitting manifests.

| Dataset root | Scenes and render contract | Created | Sole purpose later |
|---|---|---|---|
| `train-v1` | The 130 official-train scenes in `fit.txt`; training sampler, at most 512 views per scene | Stage A | Fit candidate recipes. The tiny-overfit diagnostic selects four deterministic, scene-diverse samples from this root only. |
| `development-v1` | The disjoint 15 official-train scenes in `development.txt`; validation sampler, at most 256 views per scene | Stage A | Measure generalization after each recipe-development epoch and select `checkpoints/best` by development known-class mIoU. It never supplies gradients. |
| `train-all-v1` | All 145 official training scenes; training sampler, at most 512 views per scene | Stage B | Refit the already-selected recipe for its frozen duration. It is a fresh, uniform render—not a concatenation of `train-v1` and `development-v1`. With no development selection in this refit, `checkpoints/last` is the protocol checkpoint. |
| `official-val-v1` | All 36 official validation scenes; validation sampler, at most 256 views per scene | Stage B | Report final hard-segmentation performance once, after recipe and duration are frozen. It must not influence recipe selection. |
| `calibration-fit-v1` | The 12 official-validation scenes frozen in `calibration_fit.txt`; validation sampler | Stage B | Freeze every model weight and fit only the scalar softmax temperature. |
| `calibration-evaluation-v1` | The other 24 official-validation scenes frozen in `calibration_evaluation.txt`; validation sampler | Stage B | Evaluate probability calibration on scenes that were not used to fit the temperature. It supplies neither model-weight nor temperature updates. |

`pilot` is a preliminary renderer/taxonomy/camera diagnostic and is not one of
the six server inputs. Likewise, audits and inspection images remain useful
workstation evidence but are not training data.

The chronological use is:

```text
Workstation: render train-v1 + development-v1                 (Stage A)
             render train-all-v1 + official/calibration roots (Stage B)
             validate all six and transfer them once

GPU server:  tiny overfit on 4 samples from train-v1
             train recipes on train-v1 -> select on development-v1
             freeze recipe and duration
             refit on train-all-v1 -> use checkpoints/last
             evaluate once on official-val-v1
             fit T on calibration-fit-v1
             evaluate probabilities on calibration-evaluation-v1
```

The calibration roots partition the same 36 official-validation scenes also
present in `official-val-v1`, so storing all three roots intentionally
duplicates rendered data. The full root supports one self-contained official
segmentation evaluation. The two disjoint roots enforce the no-leakage
temperature-calibration protocol. Temperature scaling cannot change the
argmax segmentation mask.

## Stage A: generate recipe-development datasets

Run both plans before starting the large render:

```bash
hm3d-semseg generate-dataset \
  --config configs/data/train.yaml \
  --local-config configs/local.yaml \
  --split-list configs/data/splits/fit.txt \
  --dry-run

hm3d-semseg generate-dataset \
  --config configs/data/validation.yaml \
  --local-config configs/local.yaml \
  --split-list configs/data/splits/development.txt \
  --dry-run
```

The expected upper bounds are 66,560 fit samples from 130 scenes and 3,840
development samples from 15 disjoint scenes. Review the output roots and storage
estimates before removing `--dry-run`.

Generate only after all preceding checks pass:

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

Generation displays two progress bars on stderr. The outer bar reports completed
and remaining scenes with elapsed time and ETA. The inner bar reports attempted
views for the current scene and counts accepted, rejected, and already-existing
samples. The stored count includes resumed manifest records. Use `--no-progress`
for non-interactive batch logs; the final JSON summary remains on stdout.

EGL/driver failures are actionable blockers, not silently skipped frames. The
rendered directory—not the HM3D source tree—is the portable handoff to training.
Keep every root-level manifest, configuration, provenance, and camera file with
its scene directories.

After validating these two roots in step 06, step 06a completes Stage B by
rendering and validating `train-all-v1`, `official-val-v1`,
`calibration-fit-v1`, and `calibration-evaluation-v1`. Keeping the stages
separate makes it possible to prove the 130/15 development pipeline first;
rendering all six before the first server transfer avoids a second handoff.

Next: [dataset format](06_dataset_format.md).
