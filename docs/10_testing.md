# 10. Testing

Testing is machine-specific: renderer/HM3D integration stays on the workstation;
the workstation training environment runs tiny-overfit and two bounded smoke
recipes before handoff; the server repeats tiny-overfit as a hardware acceptance
test; and final checkpoint inference/camera compatibility runs again on the
ObjectNav host after artifacts return.

Fast unit tests require no Habitat, HM3D, GPU, or network:

```bash
pytest -m "unit"
```

They cover strict config/precedence; official camera fixture and mismatch;
quoted descriptor/mapping parsing; no semantic-ID offset; unknown versus ignore;
contiguous IDs; masks/transforms; class zero; nearest resize; 41-channel output;
softmax; raw-logit cross-entropy; ignored metrics; hand IoU; absent classes;
ObjectNav-six; calibration; leakage; deterministic sampling; and checkpoint
round-trip. Simulated host profiles additionally cover CPU, Pascal/CUDA 12.6,
modern CUDA 13, older drivers, unsafe driver detection, and multi-GPU runtime
selection without requiring a physical GPU.

Habitat integration tests require a configured proprietary dataset:

```bash
export HM3D_SEMSEG_LOCAL_CONFIG=/absolute/path/to/configs/local.yaml
pytest -m "habitat"
```

They load annotated TEEsavR23oF, verify a nonempty semantic scene, confirm
`semantic_id=1` is descriptor object 1 (`ceiling`) with no offset, sample a
navigable point, close, and reopen. Rendering inspection is exercised by the
`inspect-scene` command because it produces reviewable artifacts.

End-to-end renderer smoke sequence:

```bash
hm3d-semseg smoke-test --local-config configs/local.yaml
pytest -m "smoke"
```

The command generates four frames, loads multiple batches, runs
forward/backward, computes metrics, saves/reloads a checkpoint, runs inference,
and writes diagnostic panels. It requires both the configured render environment
and the kernel-verified training runtime described in the installation guide.
It constructs its diagnostic settings internally and does not consume an
experiment YAML. Do not confuse it with the two bounded `_smoke.yaml` training
recipes in the pre-handoff sequence below.

Run lint separately:

```bash
ruff check .
mypy src
```

Both checks and the unit suite run automatically on Python 3.9 and 3.10 through
`.github/workflows/quality.yml`. Local quality-tool versions are reproducible via
`constraints/quality.txt`.

Skipped proprietary/GPU tests are not passes; record skips and blockers in
handoffs. On a training-only server, unit tests plus a transferred-dataset
validation and deterministic tiny-overfit run are the acceptance sequence;
Habitat and renderer smoke tests remain on the licensed rendering workstation.

The three pre-handoff training commands, their exact datasets, and acceptance
boundary are in
[workstation pre-handoff training tests](06a_server_handoff.md#2a-workstation-pre-handoff-training-tests).
They write only below `paths.runs_root` and never calibrate the model.

Next: [troubleshooting](11_troubleshooting.md).
