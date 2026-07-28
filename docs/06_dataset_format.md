# 6. Dataset format

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

Next: [training](07_training.md).

