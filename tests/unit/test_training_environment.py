from __future__ import annotations

from typing import Optional, Tuple

import pytest

from hm3d_semseg.installation.training_env import (
    GPUInfo,
    HostInfo,
    TrainingEnvironmentError,
    parse_gpu_query,
    select_profile,
)
from hm3d_semseg.utils.device import (
    DeviceCandidate,
    DeviceCompatibilityError,
    choose_device_candidate,
)

pytestmark = pytest.mark.unit


def _host(
    capability: Optional[Tuple[int, int]],
    driver_cuda: Optional[Tuple[int, int]],
    *,
    memory_mib: int = 8192,
) -> HostInfo:
    return HostInfo(
        nvidia_smi="/usr/bin/nvidia-smi",
        driver_cuda=driver_cuda,
        gpus=(
            GPUInfo(
                index=0,
                name="Test GPU",
                compute_capability=capability,
                driver_version="999.0",
                memory_mib=memory_mib,
            ),
        ),
    )


def test_parse_gpu_query() -> None:
    result = parse_gpu_query("0, NVIDIA GeForce GTX 1070, 6.1, 580.10, 8192\n")
    assert result == (
        GPUInfo(
            index=0,
            name="NVIDIA GeForce GTX 1070",
            compute_capability=(6, 1),
            driver_version="580.10",
            memory_mib=8192,
        ),
    )


def test_auto_profile_is_hardware_and_driver_aware() -> None:
    assert select_profile(HostInfo(None, None, ())).name == "cpu"
    assert select_profile(_host((6, 1), (13, 0))).name == "cu126"
    assert select_profile(_host((8, 6), (13, 0))).name == "cu130"
    assert select_profile(_host((8, 6), (12, 8))).name == "cu126"
    assert select_profile(_host((8, 6), (11, 8))).name == "cu118"


def test_cuda_13_is_rejected_for_pascal_even_when_explicit() -> None:
    with pytest.raises(TrainingEnvironmentError, match=r"requires compute capability 7\.5"):
        select_profile(_host((6, 1), (13, 0)), "cu130")


def test_failed_nvidia_probe_does_not_silently_select_cpu() -> None:
    host = HostInfo(
        nvidia_smi="/usr/bin/nvidia-smi",
        driver_cuda=None,
        gpus=(),
        probe_error="driver communication failed",
    )
    with pytest.raises(TrainingEnvironmentError, match="detection failed"):
        select_profile(host)
    assert select_profile(host, "cpu").name == "cpu"


def test_runtime_selection_prefers_working_gpu_with_more_free_memory() -> None:
    candidates = [
        DeviceCandidate(0, "busy", (8, 6), 1024, 8192, True),
        DeviceCandidate(1, "free", (8, 0), 4096, 8192, True),
        DeviceCandidate(2, "incompatible", (9, 0), 8192, 8192, False, "kernel error"),
    ]
    assert choose_device_candidate(candidates).index == 1
    with pytest.raises(DeviceCompatibilityError, match="failed a real kernel"):
        choose_device_candidate(candidates, requested_index=2)
