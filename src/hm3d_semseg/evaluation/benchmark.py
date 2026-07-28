"""Native-camera batch-1 inference efficiency benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hm3d_semseg.inference.api import SemanticSegmenter
from hm3d_semseg.models.segformer import predict
from hm3d_semseg.utils.hashing import atomic_write_json
from hm3d_semseg.utils.provenance import collect_provenance


def benchmark_inference(
    checkpoint: Path,
    output: Path,
    *,
    iterations: int = 100,
    warmup: int = 20,
    device: Optional[str] = None,
    half_precision: bool = True,
) -> Dict[str, Any]:
    """Measure synchronized end-to-end model latency at frozen camera resolution."""
    import torch

    segmenter = SemanticSegmenter.from_checkpoint(checkpoint, device=device)
    if segmenter.camera_profile is None:
        raise ValueError("Checkpoint lacks camera_profile.yaml; native resolution is unknown")
    height = segmenter.camera_profile.rgb.height
    width = segmenter.camera_profile.rgb.width
    tensor = torch.randn(1, 3, height, width, device=segmenter.device)
    use_half = half_precision and segmenter.device.startswith("cuda")
    if segmenter.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(segmenter.device)

    def synchronize() -> None:
        if segmenter.device.startswith("cuda"):
            torch.cuda.synchronize(segmenter.device)

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_half,
            ):
                predict(
                    segmenter.model,
                    tensor,
                    output_size=(height, width),
                    align_corners=segmenter.model_config.align_corners,
                    temperature=segmenter.temperature,
                )
        synchronize()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_half,
            ):
                predict(
                    segmenter.model,
                    tensor,
                    output_size=(height, width),
                    align_corners=segmenter.model_config.align_corners,
                    temperature=segmenter.temperature,
                )
            synchronize()
            timings.append(time.perf_counter() - start)
    parameters = sum(parameter.numel() for parameter in segmenter.model.parameters())
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in segmenter.model.parameters()
    )
    median = float(np.median(timings))
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "camera_profile_hash": segmenter.camera_profile.profile_hash,
        "width": width,
        "height": height,
        "batch_size": 1,
        "device": segmenter.device,
        "hardware": (
            torch.cuda.get_device_name(segmenter.device)
            if segmenter.device.startswith("cuda")
            else "CPU"
        ),
        "precision": "float16" if use_half else "float32",
        "warmup_iterations": warmup,
        "timing_iterations": iterations,
        "parameter_count": parameters,
        "model_bytes": model_bytes,
        "median_latency_seconds": median,
        "p95_latency_seconds": float(np.percentile(timings, 95)),
        "frames_per_second": 1.0 / median,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(segmenter.device))
            if segmenter.device.startswith("cuda")
            else None
        ),
        "software": collect_provenance(),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "benchmark.json", report)
    return report
