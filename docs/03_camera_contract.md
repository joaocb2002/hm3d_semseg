# 3. Camera contract

**Execution location: workstation.** The resolved camera profile becomes part
of every portable dataset and checkpoint, so the server does not need to
re-compose the ObjectNav Habitat configuration.

## Environment

Run this entire step in the render environment, not the training environment:

```bash
conda activate habitat
export PYTHONNOUSERSITE=1
cd ~/projects/hm3d-semseg
```

Before continuing, `which python` must point inside
`miniconda3/envs/habitat`, and `hm3d-semseg doctor --local-config
configs/local.yaml` must report `"ok": true`. Keep using `habitat` through scene
inspection, taxonomy auditing, and dataset generation. Switch to
`hm3d-semseg-train` only for model workflows.

## Purpose

Training images must share the ObjectNav camera's projection and extrinsics.
Raw YAML cannot resolve Hydra defaults, so the command calls the installed
Habitat 0.3.3 `habitat.get_config`. This step composes the actual ObjectNav
Hydra configuration and freezes its RGB/depth projection, sensor pose, agent
dimensions, depth convention, and look-action increment into one hashed YAML
contract. Dataset generation and checkpoints preserve that hash so training and
deployment cannot silently use different camera geometry.

This command only resolves and records configuration; it does not generate a
dataset or train a model.

```bash
hm3d-semseg resolve-camera \
  --local-config configs/local.yaml \
  --output /home/joaocb2002/hm3d-semseg-data/generated/camera/objectnav_resolved.yaml
```

## Acceptance checks

Inspect the printed profile and saved YAML before editing `configs/local.yaml`.
On the current checkout, success means:

- RGB and depth are 640×480 with 79° HFOV;
- both sensors use position `[0, 0.88, 0]` and orientation `[0, 0, 0]`;
- depth covers 0.5–5.0 m and is normalized;
- the agent is 0.88 m high with 0.18 m radius;
- look actions are supported in 30° increments, while minimum/maximum repeat
  bounds remain unknown;
- provenance records Habitat-Lab commit
  `9a1ddcb1c8b94b8555610303ec80c420b9faeb63`, version 0.3.3, and
  `fallback: false`;
- `warnings` is empty and the file reloads without a camera-profile hash error.

The printed official-2023 comparison is expected to differ: that regression
fixture is portrait 480×640, 42°, position 1.31 m, and agent 1.41/0.17 m. Never
swap width and height or choose the official fixture over the actual local
camera. The Gym, pybullet-build, and duplicate-plugin messages printed during
Habitat import are non-blocking when the command completes and emits the
profile.

`semantic: null` is also expected here: the ObjectNav source configuration does
not declare a semantic simulator sensor. Dataset generation constructs the
semantic sensor from the frozen RGB geometry so the observations stay aligned.

After every acceptance check passes, reference the immutable output in
`configs/local.yaml`; do not hand-edit the generated profile:

```yaml
camera:
  profile: /home/joaocb2002/hm3d-semseg-data/generated/camera/objectnav_resolved.yaml
  pitch_degrees: null
  require_explicit_pitch: false
  allow_mismatch: false
```

The resolved config exposes look increments but not repeat bounds. The pilot
documents level-only capture. Before a full run, inspect actual controller
behavior and add an explicit list such as the confirmed reachable angles to
`camera.pitch_degrees`. Do not infer bounds from the 30° step. Full configs set
`require_explicit_pitch: true` and fail until this decision is recorded.

The semantic sensor is constructed from RGB geometry. Depth uses its own range
but the same projection/extrinsics. Dataset and checkpoint camera hashes are
checked; `allow_mismatch` is a deliberate research override only.

If Habitat composition fails, fix the environment/config. `--raw-yaml-fallback`
is diagnostic, emits a prominent warning, records fallback provenance, and is
not acceptable for full generation.

Next: [inspect taxonomy](04_hm3d_and_taxonomy.md).
