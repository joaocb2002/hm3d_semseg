# 6. Dataset format

**Execution location: workstation.** Validate the portable dataset contract
here before transfer. The server repeats validation after copying, while the
original workstation datasets remain the recoverable source of truth.

Each version is self-describing:

```text
pilot/
├── dataset.yaml
├── resolved_config.yaml
├── provenance.json
├── camera_profile.yaml
├── manifest.jsonl
└── minival/<scene-id>/
    ├── rgb/
    ├── mask/
    ├── depth/
    └── metadata/
```

`dataset.yaml` freezes schema, seed, split, camera/taxonomy hashes, codec, and
depth. Manifest records contain stable sample/scene IDs, relative paths, sizes,
class counts, and hashes. Per-view JSON contains pose and every visible
semantic-ID decision.

Validate immediately:

```bash
hm3d-semseg validate-dataset --dataset /absolute/generated/root/pilot
```

Success means no missing/duplicate/undecodable sample, only targets 0–40 or 255,
matching RGB/mask dimensions, no scene crossing split labels, consistent
camera/taxonomy hashes, and reported unknown/ignored fractions.

The loader preserves native shape. It applies ImageNet normalization associated
with the pretrained checkpoint but no implicit square resize or label reduction.
Horizontal flips affect RGB and mask together; masks use nearest interpolation;
photometric jitter affects RGB only. Class zero remains zero.

Scene lists—not rendered frames—define fit/development/calibration partitions.
Never randomly split the manifest.

The [dataset lifecycle table](05_sampling_and_generation.md#dataset-lifecycle-why-six-roots-exist)
is the source of truth for which root is allowed at each later stage. In
particular, the tiny-overfit diagnostic uses four samples from `train-v1`,
recipe selection uses `development-v1`, and neither is a substitute for the
fresh 145-scene `train-all-v1` final-refit root.

Next: [prepare the workstation-to-server handoff](06a_server_handoff.md). It
explains which complete dataset directories move, how to verify them, how to
recreate the host-specific environment, and which final artifacts return.
