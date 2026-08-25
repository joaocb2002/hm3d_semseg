# 9. Inference and ObjectNav integration

Return and checksum the complete final run as described in the
[server handoff guide](server_handoff.md). Use the calibrated checkpoint
directory as the deployment unit; `model.safetensors` alone omits the model
configuration, project metadata, camera contract, and fitted temperature.

Install only the inference dependencies into the downstream environment; the
host-aware training installer and training-only plotting/TensorBoard packages
are not required:

```bash
cd /absolute/path/to/hm3d-semseg
python -m pip install -e ".[inference]"
```

Single-image inference preserves the input aspect ratio:

```bash
hm3d-semseg infer \
  --checkpoint /absolute/checkpoint \
  --image /absolute/path/to/rgb.png \
  --output /absolute/path/to/inference
```

It saves lossless class IDs, color mask, RGB overlay, confidence, entropy, and
metadata. Add `--save-probabilities` only when the large float32 `[41,H,W]`
tensor is needed. Softmax is across class dimension; probabilities sum to one at
every pixel.

Stable Python API:

```python
from pathlib import Path
import numpy as np
from hm3d_semseg import inference

segmenter = inference.SemanticSegmenter.from_checkpoint(
    Path("/absolute/checkpoint"), device="cuda"
)
result = segmenter(rgb_uint8)
probabilities = result["probabilities"]  # [41, H, W]
labels = result["labels"]                # [H, W]
```

Before ObjectNav use, resolve its runtime camera and check:

```python
from hm3d_semseg.camera import resolve_camera_profile

runtime = resolve_camera_profile(Path("/absolute/objectnav.yaml"))
segmenter.assert_camera(runtime)
```

Do not import `object-nav-v2` into this repository. In the downstream project,
import the installed `hm3d_semseg` package and pass RGB observations directly.
Start with the repository commit recorded in the returned run's provenance;
upgrading inference code later requires a regression test against the frozen
checkpoint. Preserve the complete research run separately even if deployment
omits the resume-only `training_state.pt` file.
Use the same RGB channel order, ImageNet mean/std, no label reduction, no square
warp, and the checkpoint's temperature. Output class `sofa` corresponds to
ObjectNav couch, `plant` to potted plant, and `tv_monitor` to TV.

Future rollout-manifest evaluation should remain secondary: policy behavior
changes the observation distribution.

Next: [testing](10_testing.md).
