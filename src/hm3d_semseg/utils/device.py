"""Safe and efficient PyTorch device selection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hm3d_semseg.exceptions import HM3DSemsegError, OptionalDependencyError


class DeviceCompatibilityError(HM3DSemsegError):
    """Raised when CUDA is visible but cannot execute this PyTorch build."""


@dataclass(frozen=True)
class DeviceCandidate:
    """One CUDA device and the outcome of a real kernel probe."""

    index: int
    name: str
    compute_capability: Tuple[int, int]
    free_memory_bytes: Optional[int]
    total_memory_bytes: Optional[int]
    kernel_ok: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class DeviceSelection:
    """Resolved runtime device with evidence for the choice."""

    device: str
    reason: str
    compiled_architectures: Tuple[str, ...]
    candidates: Tuple[DeviceCandidate, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _probe_candidate(torch: Any, index: int) -> DeviceCandidate:
    name = str(torch.cuda.get_device_name(index))
    capability = tuple(torch.cuda.get_device_capability(index))
    free_memory: Optional[int] = None
    total_memory: Optional[int] = None
    try:
        free, total = torch.cuda.mem_get_info(index)
        free_memory, total_memory = int(free), int(total)
    except Exception:
        pass
    try:
        value = torch.ones((16, 16), device=f"cuda:{index}")
        result = value @ value
        torch.cuda.synchronize(index)
        if float(result[0, 0].item()) != 16.0:
            raise RuntimeError("CUDA verification produced an incorrect result")
        return DeviceCandidate(
            index=index,
            name=name,
            compute_capability=(int(capability[0]), int(capability[1])),
            free_memory_bytes=free_memory,
            total_memory_bytes=total_memory,
            kernel_ok=True,
        )
    except Exception as error:
        return DeviceCandidate(
            index=index,
            name=name,
            compute_capability=(int(capability[0]), int(capability[1])),
            free_memory_bytes=free_memory,
            total_memory_bytes=total_memory,
            kernel_ok=False,
            error=repr(error),
        )


def choose_device_candidate(
    candidates: Sequence[DeviceCandidate],
    requested_index: Optional[int] = None,
) -> DeviceCandidate:
    """Choose a working GPU, preferring available memory then architecture."""
    if requested_index is not None:
        matching = [candidate for candidate in candidates if candidate.index == requested_index]
        if not matching:
            raise DeviceCompatibilityError(
                f"CUDA device {requested_index} does not exist or is not visible."
            )
        selected = matching[0]
        if not selected.kernel_ok:
            raise DeviceCompatibilityError(
                f"CUDA device {requested_index} ({selected.name}) failed a real kernel "
                f"probe: {selected.error}"
            )
        return selected
    working = [candidate for candidate in candidates if candidate.kernel_ok]
    if not working:
        details = "; ".join(
            f"cuda:{candidate.index} {candidate.name} "
            f"CC {candidate.compute_capability[0]}.{candidate.compute_capability[1]}: "
            f"{candidate.error}"
            for candidate in candidates
        )
        raise DeviceCompatibilityError(
            "CUDA is reported as available, but no visible GPU can execute this "
            f"PyTorch build. {details}"
        )
    return max(
        working,
        key=lambda candidate: (
            candidate.free_memory_bytes or -1,
            candidate.compute_capability,
        ),
    )


def select_torch_device(requested: Optional[str] = None) -> DeviceSelection:
    """Resolve CPU or a kernel-verified CUDA device for a long-running workflow."""
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyError(
            "PyTorch is required. Run `hm3d-semseg install-training-env` first."
        ) from error
    value = (requested or "auto").lower()
    if value == "cpu":
        return DeviceSelection(
            device="cpu",
            reason="CPU was requested explicitly.",
            compiled_architectures=(),
            candidates=(),
        )
    match = re.fullmatch(r"cuda(?::(\d+))?", value)
    if value != "auto" and match is None:
        raise DeviceCompatibilityError(
            f"Invalid device {requested!r}; use auto, cpu, cuda, or cuda:N."
        )
    if not torch.cuda.is_available():
        if value == "auto":
            from hm3d_semseg.installation.training_env import detect_host

            host = detect_host()
            if host.probe_error:
                raise DeviceCompatibilityError(
                    "CUDA is unavailable and NVIDIA driver detection failed: "
                    f"{host.probe_error}. Use an explicit `cpu` device only if CPU "
                    "execution is intentional."
                )
            if host.gpus:
                raise DeviceCompatibilityError(
                    "NVIDIA hardware is visible, but this PyTorch runtime has no "
                    "usable CUDA support. Run `hm3d-semseg install-training-env` or "
                    "request `cpu` explicitly."
                )
            return DeviceSelection(
                device="cpu",
                reason="No CUDA runtime is available; selected CPU.",
                compiled_architectures=(),
                candidates=(),
            )
        raise DeviceCompatibilityError(
            f"{value} was requested, but torch.cuda.is_available() is false."
        )
    architectures = tuple(torch.cuda.get_arch_list())
    candidates: List[DeviceCandidate] = []
    for index in range(torch.cuda.device_count()):
        try:
            candidates.append(_probe_candidate(torch, index))
        except Exception as error:
            candidates.append(
                DeviceCandidate(
                    index=index,
                    name=f"CUDA device {index}",
                    compute_capability=(0, 0),
                    free_memory_bytes=None,
                    total_memory_bytes=None,
                    kernel_ok=False,
                    error=repr(error),
                )
            )
    requested_index = (
        int(match.group(1)) if match is not None and match.group(1) is not None else None
    )
    if value == "cuda":
        requested_index = 0
    try:
        selected = choose_device_candidate(
            candidates,
            requested_index=None if value == "auto" else requested_index,
        )
    except DeviceCompatibilityError as error:
        compiled = ", ".join(architectures) or "none reported"
        raise DeviceCompatibilityError(
            f"{error} Compiled CUDA architectures: {compiled}. Run "
            "`hm3d-semseg install-training-env` to select a compatible wheel."
        ) from error
    torch.cuda.set_device(selected.index)
    reason = (
        "Auto-selected the working GPU with the most currently free memory."
        if value == "auto"
        else f"Validated requested CUDA device {selected.index} with a real kernel."
    )
    return DeviceSelection(
        device=f"cuda:{selected.index}",
        reason=reason,
        compiled_architectures=architectures,
        candidates=tuple(candidates),
    )
