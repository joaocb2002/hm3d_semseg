"""Plan and apply a PyTorch installation matched to the current host."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from hm3d_semseg.exceptions import HM3DSemsegError

Version = Tuple[int, int]


class TrainingEnvironmentError(HM3DSemsegError):
    """Raised when a safe training environment cannot be selected or installed."""


@dataclass(frozen=True)
class GPUInfo:
    """Hardware facts reported by NVIDIA's driver utility."""

    index: int
    name: str
    compute_capability: Optional[Version]
    driver_version: str
    memory_mib: int


@dataclass(frozen=True)
class HostInfo:
    """Host facts used for profile selection."""

    nvidia_smi: Optional[str]
    driver_cuda: Optional[Version]
    gpus: Tuple[GPUInfo, ...]
    probe_error: Optional[str] = None


@dataclass(frozen=True)
class TorchProfile:
    """One tested pairing of PyTorch, torchvision, and CUDA runtime."""

    name: str
    torch_version: str
    torchvision_version: str
    index_url: str
    cuda_runtime: Optional[Version]
    minimum_driver_cuda: Optional[Version]
    minimum_compute_capability: Optional[Version]
    rationale: str


PROFILES: Dict[str, TorchProfile] = {
    "cpu": TorchProfile(
        name="cpu",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        index_url="https://download.pytorch.org/whl/cpu",
        cuda_runtime=None,
        minimum_driver_cuda=None,
        minimum_compute_capability=None,
        rationale="No usable NVIDIA GPU was detected, or CPU was requested explicitly.",
    ),
    "cu118": TorchProfile(
        name="cu118",
        torch_version="2.7.1",
        torchvision_version="0.22.1",
        index_url="https://download.pytorch.org/whl/cu118",
        cuda_runtime=(11, 8),
        minimum_driver_cuda=(11, 8),
        minimum_compute_capability=(6, 0),
        rationale="Compatibility profile for an older NVIDIA driver.",
    ),
    "cu126": TorchProfile(
        name="cu126",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        index_url="https://download.pytorch.org/whl/cu126",
        cuda_runtime=(12, 6),
        minimum_driver_cuda=(12, 6),
        minimum_compute_capability=(6, 0),
        rationale="CUDA 12.6 retains Pascal support and is the safe profile for CC 6.x GPUs.",
    ),
    "cu130": TorchProfile(
        name="cu130",
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        index_url="https://download.pytorch.org/whl/cu130",
        cuda_runtime=(13, 0),
        minimum_driver_cuda=(13, 0),
        minimum_compute_capability=(7, 5),
        rationale="Newest profile for a CUDA 13 driver and a PyTorch-supported modern GPU.",
    ),
}


def _parse_version(value: str) -> Optional[Version]:
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_gpu_query(output: str) -> Tuple[GPUInfo, ...]:
    """Parse the stable CSV query emitted by ``nvidia-smi``."""
    gpus: List[GPUInfo] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise TrainingEnvironmentError(
                f"Could not parse nvidia-smi GPU row {line_number}: {line!r}"
            )
        capability = _parse_version(fields[2])
        try:
            gpus.append(
                GPUInfo(
                    index=int(fields[0]),
                    name=fields[1],
                    compute_capability=capability,
                    driver_version=fields[3],
                    memory_mib=int(float(fields[4])),
                )
            )
        except ValueError as error:
            raise TrainingEnvironmentError(
                f"Could not parse nvidia-smi GPU row {line_number}: {line!r}"
            ) from error
    return tuple(gpus)


def detect_host() -> HostInfo:
    """Read NVIDIA hardware and maximum driver CUDA without importing PyTorch."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return HostInfo(nvidia_smi=None, driver_cuda=None, gpus=())
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        detail = query.stderr.strip() or query.stdout.strip() or "unknown nvidia-smi error"
        return HostInfo(
            nvidia_smi=executable,
            driver_cuda=None,
            gpus=(),
            probe_error=detail,
        )
    try:
        gpus = parse_gpu_query(query.stdout)
    except TrainingEnvironmentError as error:
        return HostInfo(
            nvidia_smi=executable,
            driver_cuda=None,
            gpus=(),
            probe_error=str(error),
        )
    summary = subprocess.run(
        [executable],
        check=False,
        capture_output=True,
        text=True,
    )
    summary_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary.stdout)
    driver_cuda = _parse_version(summary_match.group(1) if summary_match else "")
    if gpus and driver_cuda is None:
        return HostInfo(
            nvidia_smi=executable,
            driver_cuda=None,
            gpus=gpus,
            probe_error="nvidia-smi did not report the driver's maximum CUDA version",
        )
    return HostInfo(nvidia_smi=executable, driver_cuda=driver_cuda, gpus=gpus)


def _best_gpu(gpus: Sequence[GPUInfo]) -> GPUInfo:
    def rank(gpu: GPUInfo) -> Tuple[int, Version]:
        return gpu.memory_mib, gpu.compute_capability or (0, 0)

    return max(gpus, key=rank)


def select_profile(
    host: HostInfo,
    requested: str = "auto",
    *,
    allow_unsupported_host: bool = False,
) -> TorchProfile:
    """Select and validate the most capable safe profile for a host."""
    requested = requested.lower()
    if requested not in {"auto", *PROFILES}:
        raise TrainingEnvironmentError(
            f"Unknown profile {requested!r}; choose auto, {', '.join(sorted(PROFILES))}"
        )
    if requested == "cpu":
        return PROFILES["cpu"]
    if host.probe_error and not allow_unsupported_host:
        raise TrainingEnvironmentError(
            "NVIDIA hardware detection failed, so auto-installation is unsafe: "
            f"{host.probe_error}. Repair nvidia-smi, request --profile cpu, or use an "
            "explicit CUDA profile with --allow-unsupported-host when provisioning "
            "for a different machine."
        )
    if requested == "auto":
        if not host.gpus:
            return PROFILES["cpu"]
        if host.driver_cuda is None:
            raise TrainingEnvironmentError(
                "The NVIDIA driver CUDA compatibility version could not be detected."
            )
        gpu = _best_gpu(host.gpus)
        capability = gpu.compute_capability
        if capability is None:
            raise TrainingEnvironmentError(
                f"Compute capability could not be detected for {gpu.name}."
            )
        if capability < (6, 0):
            raise TrainingEnvironmentError(
                f"{gpu.name} has compute capability {capability[0]}.{capability[1]}, "
                "below the tested CUDA profile minimum 6.0. Use --profile cpu."
            )
        if host.driver_cuda >= (13, 0) and capability >= (7, 5):
            return PROFILES["cu130"]
        if host.driver_cuda >= (12, 6):
            return PROFILES["cu126"]
        if host.driver_cuda >= (11, 8):
            return PROFILES["cu118"]
        raise TrainingEnvironmentError(
            "The NVIDIA driver supports only CUDA "
            f"{host.driver_cuda[0]}.{host.driver_cuda[1]}; update it to support at "
            "least CUDA 11.8, or use --profile cpu."
        )
    profile = PROFILES[requested]
    if allow_unsupported_host:
        return profile
    if not host.gpus:
        raise TrainingEnvironmentError(
            f"Profile {requested} requires an NVIDIA GPU, but none was detected."
        )
    gpu = _best_gpu(host.gpus)
    if profile.minimum_compute_capability is not None and (
        gpu.compute_capability is None
        or gpu.compute_capability < profile.minimum_compute_capability
    ):
        required = profile.minimum_compute_capability
        found = gpu.compute_capability
        if found is None:
            raise TrainingEnvironmentError(
                f"Profile {requested} requires a known compatible compute capability."
            )
        raise TrainingEnvironmentError(
            f"Profile {requested} requires compute capability "
            f"{required[0]}.{required[1]} or newer; {gpu.name} reports "
            f"{found[0]}.{found[1]}."
        )
    if profile.minimum_driver_cuda is not None and (
        host.driver_cuda is None or host.driver_cuda < profile.minimum_driver_cuda
    ):
        required = profile.minimum_driver_cuda
        raise TrainingEnvironmentError(
            f"Profile {requested} requires a driver supporting CUDA "
            f"{required[0]}.{required[1]} or newer."
        )
    return profile


_RUNTIME_PROBE = r"""
import json
report = {"installed": False}
try:
    import torch
    report.update({
        "installed": True,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torchvision": None,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "compiled_architectures": (
            torch.cuda.get_arch_list() if torch.version.cuda is not None else []
        ),
        "devices": [],
    })
    cpu_value = torch.ones((16, 16))
    cpu_result = cpu_value @ cpu_value
    report["cpu_kernel_ok"] = bool(cpu_result[0, 0].item() == 16.0)
    try:
        import torchvision
        report["torchvision"] = torchvision.__version__
    except Exception as error:
        report["torchvision_error"] = repr(error)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device = {"index": index}
            try:
                device.update({
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                })
                value = torch.ones((16, 16), device=f"cuda:{index}")
                result = value @ value
                torch.cuda.synchronize(index)
                device.update({"kernel_ok": bool(result[0, 0].item() == 16.0)})
            except Exception as error:
                device.update({"kernel_ok": False, "error": repr(error)})
            report["devices"].append(device)
except Exception as error:
    report["error"] = repr(error)
print(json.dumps(report))
"""


def inspect_torch_runtime() -> Dict[str, Any]:
    """Inspect the active interpreter in an isolated, user-site-disabled process."""
    result = subprocess.run(
        [sys.executable, "-c", _RUNTIME_PROBE],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_environment(),
    )
    try:
        report = cast(Dict[str, Any], json.loads(result.stdout.strip()))
    except json.JSONDecodeError:
        report = {
            "installed": False,
            "error": "Torch inspection did not return JSON",
            "stdout": result.stdout[-2000:],
        }
    if result.stderr.strip():
        report["stderr"] = result.stderr[-4000:].strip()
    report["returncode"] = result.returncode
    return report


def _profile_matches_runtime(
    profile: TorchProfile,
    runtime: Dict[str, Any],
    *,
    require_cuda_kernel: bool = True,
) -> bool:
    if (
        not runtime.get("installed")
        or runtime.get("returncode") != 0
        or not runtime.get("cpu_kernel_ok")
    ):
        return False
    if not str(runtime.get("torch", "")).startswith(profile.torch_version):
        return False
    if not str(runtime.get("torchvision", "")).startswith(profile.torchvision_version):
        return False
    cuda_build = _parse_version(str(runtime.get("cuda_build") or ""))
    if cuda_build != profile.cuda_runtime:
        return False
    if profile.cuda_runtime is None:
        return True
    if not require_cuda_kernel:
        return True
    return any(device.get("kernel_ok") for device in runtime.get("devices", []))


def _clean_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _validate_project_root(project_root: Path) -> Path:
    root = project_root.expanduser().resolve()
    metadata = root / "pyproject.toml"
    if not metadata.is_file():
        raise TrainingEnvironmentError(f"No pyproject.toml found under {root}")
    text = metadata.read_text(encoding="utf-8")
    if not re.search(r'^name\s*=\s*["\']hm3d-semseg["\']', text, re.MULTILINE):
        raise TrainingEnvironmentError(f"{root} is not the hm3d-semseg project root")
    return root


def _command_plan(
    profile: TorchProfile,
    root: Path,
    *,
    with_dev: bool,
    install_torch: bool,
    reinstall_torch: bool,
    run_tests: bool,
) -> List[List[str]]:
    torch_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    ]
    if reinstall_torch:
        torch_command.append("--force-reinstall")
    torch_command.extend(
        [
            f"torch=={profile.torch_version}",
            f"torchvision=={profile.torchvision_version}",
            "--index-url",
            profile.index_url,
        ]
    )
    extras = "dev,train" if with_dev else "train"
    commands: List[List[str]] = []
    if install_torch:
        commands.append(torch_command)
    commands.extend(
        [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                f"{root}[{extras}]",
            ],
            [sys.executable, "-m", "pip", "check"],
        ]
    )
    if run_tests:
        commands.append([sys.executable, "-m", "pytest", "-m", "unit"])
    return commands


def install_training_environment(
    project_root: Path,
    *,
    profile_name: str = "auto",
    apply: bool = False,
    with_dev: bool = True,
    force_torch: bool = False,
    run_tests: bool = False,
    allow_unsupported_host: bool = False,
    host: Optional[HostInfo] = None,
) -> Dict[str, Any]:
    """Return a host-specific plan and optionally execute and verify it."""
    if sys.version_info < (3, 10):
        raise TrainingEnvironmentError(
            "The managed training profiles require Python 3.10 or newer. Create "
            "`conda create -n hm3d-semseg-train python=3.10` and rerun there."
        )
    root = _validate_project_root(project_root)
    detected = host or detect_host()
    selected = select_profile(
        detected,
        profile_name,
        allow_unsupported_host=allow_unsupported_host,
    )
    before = inspect_torch_runtime()
    require_cuda_kernel = not (allow_unsupported_host and selected.cuda_runtime is not None)
    matches = _profile_matches_runtime(
        selected,
        before,
        require_cuda_kernel=require_cuda_kernel,
    )
    install_torch = force_torch or not matches
    reinstall_torch = force_torch or (bool(before.get("installed")) and not matches)
    commands = _command_plan(
        selected,
        root,
        with_dev=with_dev,
        install_torch=install_torch,
        reinstall_torch=reinstall_torch,
        run_tests=run_tests,
    )
    report: Dict[str, Any] = {
        "applied": False,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "isolated_environment": sys.prefix != sys.base_prefix,
            "user_site_disabled_for_commands": True,
        },
        "host": asdict(detected),
        "selected_profile": asdict(selected),
        "existing_runtime": before,
        "existing_runtime_matches": matches,
        "cuda_kernel_verification_required": require_cuda_kernel,
        "torch_action": (
            "keep verified matching runtime"
            if not install_torch
            else "force reinstall selected runtime"
            if reinstall_torch
            else "install selected runtime"
        ),
        "commands": commands,
        "next_step": (
            "Review this plan, then rerun with --apply."
            if not apply
            else "Installation commands will now run."
        ),
    }
    if not apply:
        return report
    completed: List[Dict[str, Any]] = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                env=_clean_environment(),
            )
        except subprocess.CalledProcessError as error:
            raise TrainingEnvironmentError(
                f"Installation command failed with exit code {error.returncode}: "
                + " ".join(command)
            ) from error
        completed.append({"command": command, "returncode": result.returncode})
    after = inspect_torch_runtime()
    if not _profile_matches_runtime(
        selected,
        after,
        require_cuda_kernel=require_cuda_kernel,
    ):
        raise TrainingEnvironmentError(
            "The installation completed, but the selected PyTorch runtime failed "
            "version or real-kernel verification. Do not start a long training run. "
            f"Verification report: {json.dumps(after, sort_keys=True, default=str)}"
        )
    report.update(
        {
            "applied": True,
            "completed": completed,
            "verified_runtime": after,
            "next_step": "The training environment is ready.",
        }
    )
    return report
