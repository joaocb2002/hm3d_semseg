# Reference: testing and quality gates

Testing is machine-specific. Renderer/HM3D integration remains on the licensed
workstation; the workstation training environment runs tiny-overfit and two
bounded smoke recipes; the server repeats tiny overfit as a hardware acceptance
test; and the returned checkpoint runs inference/camera checks on the ObjectNav
host.

## Fast source checks

These require no Habitat, HM3D, GPU, or network:

```bash
python -m pytest -m unit
ruff check .
mypy src tests
```

They cover strict configuration and precedence, camera mismatch, descriptor and
mapping parsing, semantic-ID conversion, unknown versus ignore, transforms,
nearest mask resize, 41-channel output, raw-logit cross-entropy, confusion
metrics, absent classes, ObjectNav-six metrics, calibration leakage,
deterministic sampling, checkpoint round trips, and simulated host profiles.

## Habitat integration on the workstation

Generic form:

```bash
export HM3D_SEMSEG_LOCAL_CONFIG=/path/to/hm3d-semseg/configs/local.yaml
python -m pytest -m habitat
```

Current workstation form:

```bash
export HM3D_SEMSEG_LOCAL_CONFIG=/home/joaocb2002/projects/hm3d-semseg/configs/local.yaml
python -m pytest -m habitat
```

These tests load annotated `TEEsavR23oF`, verify a nonempty semantic scene,
confirm there is no semantic-ID offset, sample a navigable point, close, and
reopen the simulator. Skips caused by missing proprietary assets are not
passes.

## End-to-end renderer smoke test

```bash
conda activate habitat
export PYTHONNOUSERSITE=1
cd /home/joaocb2002/projects/hm3d-semseg

hm3d-semseg smoke-test --local-config configs/local.yaml
python -m pytest -m smoke
```

The CLI diagnostic generates four frames, loads batches, runs forward/backward,
computes metrics, saves/reloads a checkpoint, and performs inference. It is not
the same as the checked `_smoke.yaml` training experiments.

## Scientific workflow gates

| Gate | Machine | Required evidence |
|---|---|---|
| Renderer/taxonomy | Workstation `habitat` env | Camera, scene inspection, audits, pilot, and six roots validate. |
| Training mechanics | Workstation training env | Tiny overfit memorizes; the ADE20K-recipe bounded smoke run completes. |
| Server acceptance | GPU server | Exact SHA/data/model snapshot; unit/install tests; transferred roots validate; tiny overfit succeeds on CUDA. |
| Recipe development | GPU server | The full ADE20K-style run produces a complete held-out development report. |
| Final protocol | GPU server | Frozen recipe refits on `train-all-v1`; official hard-segmentation evaluation completes. Calibration may follow later. |
| Deployment | Workstation/ObjectNav host | Returned checksums, inference, camera assertion, and target-GPU benchmark pass. |

Training-only servers do not need Habitat tests. Workstations without the
server GPU do not need to reproduce server throughput. Preserve failures,
skips, package versions, GPU identity, and Git SHA in handoff records.

Return to the [execution guide](README.md).
