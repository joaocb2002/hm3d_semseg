# 10. Return artifacts and integrate with ObjectNav

This stage starts on the workstation after step 9 seals the final server run.
It copies artifacts through the data-transfer channel, keeps source through Git,
verifies the selected deployment checkpoint, and then loads it from the Habitat/ObjectNav
environment. The trained model is never committed to this repository.

## 10.1 Copy the complete final run back

Run from the workstation. Generic resumable command:

```bash
rsync -a \
  --partial \
  --info=progress2 \
  --checksum \
  USER@SERVER:/server/runs/FINAL_RUN/ \
  /local/hm3d-semseg-data/runs/server/FINAL_RUN/
```

Planned current command, using `knuth`'s Conda-installed remote `rsync`:

```bash
rsync -a \
  --partial \
  --info=progress2 \
  --checksum \
  --rsync-path=/workspace/miniconda/envs/hm3d-transfer/bin/rsync \
  joao_branco@knuth:/workspace/runs/segformer_b2_final/ \
  /home/joaocb2002/hm3d-semseg-data/runs/server/segformer_b2_final/
```

If training allocated a numeric suffix, use that exact server directory on
both sides. Do not add `--delete`. Rerun the command once; a near-zero second
transfer confirms identity. Keep the server copy until local verification,
inference, and backup all succeed.

The intended local separation is:

```text
/home/joaocb2002/projects/hm3d-semseg/                    # Git source
/home/joaocb2002/hm3d-semseg-data/generated/              # original datasets
/home/joaocb2002/hm3d-semseg-data/runs/server/segformer_b2_final/  # planned returned run
```

## 10.2 Verify checksums and provenance locally

Generic verification:

```bash
cd /local/hm3d-semseg-data/runs/server/FINAL_RUN
sha256sum -c deployment.sha256
```

Planned current verification:

```bash
cd /home/joaocb2002/hm3d-semseg-data/runs/server/segformer_b2_final
sha256sum -c deployment.sha256
```

Then inspect `provenance/provenance.json`, `provenance/resolved_config.yaml`,
`records/run_summary.json`, `records/metrics_summary.json`, official and
optional calibration evaluation summaries, and the benchmark. The Git SHA,
camera hash, model revision, and dataset
identities must match the accepted server record. If calibration was performed,
the temperature must also match.

On the workstation source repository, first ensure local work is safe:

```bash
cd /home/joaocb2002/projects/hm3d-semseg
git status --short
git rev-parse HEAD
```

Use the source revision recorded by the returned run. If intentional later
source changes exist, preserve them in Git rather than overwriting them. A
checkpoint remains reproducible only with its recorded revision; a later
inference-code upgrade needs a regression comparison against that revision.

## 10.3 Install only the inference dependencies

Generic command:

```bash
conda activate OBJECTNAV_ENVIRONMENT
cd /path/to/hm3d-semseg
python -m pip install -e ".[inference]"
```

Current workstation form:

```bash
conda activate habitat
cd /home/joaocb2002/projects/hm3d-semseg
python -m pip install -e ".[inference]"
```

This must not replace Habitat, Habitat-Sim, CUDA, or the environment's PyTorch.
The server training environment is not copied back.

## 10.4 Run a local single-image acceptance test

Generic command:

```bash
hm3d-semseg infer \
  --checkpoint /local/runs/server/FINAL_RUN/checkpoints/DEPLOYMENT_CHECKPOINT \
  --image /path/to/representative/rgb.png \
  --output /local/inference-check
```

Planned current command using an existing inspected HM3D view:

```bash
hm3d-semseg infer \
  --checkpoint /home/joaocb2002/hm3d-semseg-data/runs/server/segformer_b2_final/checkpoints/last \
  --image /home/joaocb2002/hm3d-semseg-data/generated/inspection/TEEsavR23oF/view_000_rgb.png \
  --output /home/joaocb2002/hm3d-semseg-data/runs/server/segformer_b2_final/inference-check
```

Inspect class IDs, color mask, RGB overlay, confidence, entropy, and metadata.
The output preserves input aspect ratio. Add `--save-probabilities` only when a
large float32 `[41, H, W]` tensor is actually needed.

Use `checkpoints/last` while calibration is deferred. Substitute
`checkpoints/calibrated` only after its disjoint calibration evaluation is
accepted.

## 10.5 Load the complete deployment checkpoint through Python

Generic API:

```python
from pathlib import Path
from hm3d_semseg import inference

segmenter = inference.SemanticSegmenter.from_checkpoint(
    Path("/local/runs/server/FINAL_RUN/checkpoints/DEPLOYMENT_CHECKPOINT"),
    device="cuda",
)
result = segmenter(rgb_uint8)
probabilities = result["probabilities"]  # [41, H, W]
labels = result["labels"]                # [H, W]
```

Current planned checkpoint path:

```python
from pathlib import Path
from hm3d_semseg import inference

segmenter = inference.SemanticSegmenter.from_checkpoint(
    Path(
        "/home/joaocb2002/hm3d-semseg-data/runs/server/"
        "segformer_b2_final/checkpoints/last"
    ),
    device="cuda",
)
result = segmenter(rgb_uint8)
probabilities = result["probabilities"]  # [41, H, W]
labels = result["labels"]                # [H, W]
```

The complete `last` or `calibrated` directory is the deployment unit. Copying only
`model.safetensors` would omit model/project metadata, label definitions,
camera/resize contract, and any fitted temperature.

## 10.6 Enforce camera compatibility in ObjectNav

Generic check:

```python
from pathlib import Path
from hm3d_semseg.camera import resolve_camera_profile

runtime_camera = resolve_camera_profile(Path("/path/to/objectnav.yaml"))
segmenter.assert_camera(runtime_camera)
```

Current ObjectNav configuration path:

```python
from pathlib import Path
from hm3d_semseg.camera import resolve_camera_profile

runtime_camera = resolve_camera_profile(
    Path(
        "/home/joaocb2002/projects/habitat-lab/habitat-lab/habitat/"
        "config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    )
)
segmenter.assert_camera(runtime_camera)
```

Integrate from the downstream ObjectNav project by importing the installed
`hm3d_semseg` package. Do not add `object-nav-v2` to this repository or modify
sibling repositories from here. Pass RGB observations in their original channel
order and geometry; the package applies the recorded ImageNet normalization and
temperature. Model class `sofa` maps to ObjectNav couch, `plant` to potted
plant, and `tv_monitor` to TV.

Finally benchmark inference on the actual ObjectNav GPU and record native
resolution, precision, latency distribution, FPS, and peak memory. Server
benchmark numbers describe `knuth`, not deployment performance.

Pipeline complete. Use the [testing reference](reference_testing.md) for
regression gates and the [troubleshooting reference](reference_troubleshooting.md)
when an environment, camera, or checkpoint check fails.
