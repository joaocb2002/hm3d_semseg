# 4. HM3D and taxonomy

Semantic observations contain scene-specific object IDs, not MPCAT40 IDs. The
pipeline is:

```text
rendered ID -> SemanticObject.semantic_id -> raw HM3D category
-> exact-normalized Matterport row -> MPCAT40/unknown/ignore -> model ID
```

Normalization is documented and limited to Unicode NFC, trim, case-fold, and
whitespace collapse. There is no fuzzy matching. Quoted punctuation is parsed
with CSV/TSV readers.

Model ID 0 is learnable `unknown`; IDs 1–40 exactly equal authoritative
`mpcat40index`; 255 is ignored ground truth and has no logit. The conservative
policy maps authoritative known rows to 1–40, unlabeled/unknown rows to model
unknown, and remove/void/missing/unmapped failures to ignore. Change policies
only in config and audit the consequences.

First inspect an annotated real scene:

```bash
hm3d-semseg inspect-scene \
  --local-config configs/local.yaml \
  --split minival \
  --scene-id 00800-TEEsavR23oF \
  --num-views 4 \
  --output /absolute/generated/root/inspection/TEEsavR23oF
```

Success means RGB, raw-ID colors, lossless class masks, color masks, depth,
overlays, histograms, and semantic-ID decisions are meaningful and aligned.

Then audit all four annotated minival scenes:

```bash
hm3d-semseg audit-taxonomy \
  --local-config configs/local.yaml \
  --split minival \
  --output /absolute/generated/root/audits/minival
```

Review `raw_label_decisions`, zero-support classes, ObjectNav-six support, and
ignored/unknown policy. Before full training repeat with `--split train` and
`--split val`; freeze the mapping asset hash and policy afterwards.

Next: [sampling and generation](05_sampling_and_generation.md).

