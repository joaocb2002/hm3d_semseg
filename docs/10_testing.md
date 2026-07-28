# 10. Testing

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

Smoke sequence:

```bash
hm3d-semseg smoke-test --local-config configs/local.yaml
pytest -m "smoke"
```

The command generates four frames, loads multiple batches, runs
forward/backward, computes metrics, saves/reloads a checkpoint, runs inference,
and writes diagnostic panels. It requires both the configured render environment
and the kernel-verified training runtime described in the installation guide.

Run lint separately:

```bash
ruff check .
```

Skipped proprietary/GPU tests are not passes; record skips and blockers in
handoffs.

Next: [troubleshooting](11_troubleshooting.md).
