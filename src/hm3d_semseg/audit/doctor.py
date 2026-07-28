"""Read-only installation and path diagnostics."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.scenes.discovery import discover_scenes
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.provenance import collect_provenance


def doctor(config: ProjectConfig) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    output_roots = {"generated_data_root", "runs_root", "cache_root"}
    for name, value in vars(config.paths).items():
        required = name in {
            "habitat_lab_root",
            "hm3d_root",
            "scene_dataset_config",
            "objectnav_config",
            "taxonomy_mapping",
            "generated_data_root",
            "runs_root",
            "cache_root",
        }
        exists = value is not None and value.exists()
        existing_parent = value.parent if value is not None else None
        while (
            existing_parent is not None
            and not existing_parent.exists()
            and existing_parent != existing_parent.parent
        ):
            existing_parent = existing_parent.parent
        parent_writable = (
            existing_parent is not None
            and existing_parent.exists()
            and os.access(existing_parent, os.W_OK)
        )
        checks[f"path:{name}"] = {
            "ok": bool(exists or (name in output_roots and parent_writable)),
            "required": required,
            "value": str(value) if value is not None else None,
            "exists": bool(exists),
            "parent_writable": bool(parent_writable),
        }
    required_imports = {
        "numpy",
        "PIL",
        "yaml",
        "typer",
        "habitat",
        "habitat_sim",
        "torch",
    }
    for module in (
        "numpy",
        "PIL",
        "yaml",
        "typer",
        "habitat",
        "habitat_sim",
        "torch",
        "transformers",
    ):
        try:
            imported = importlib.import_module(module)
            checks[f"import:{module}"] = {
                "ok": True,
                "required": module in required_imports,
                "version": getattr(imported, "__version__", None),
            }
        except Exception as error:
            checks[f"import:{module}"] = {
                "ok": False,
                "required": module in required_imports,
                "error": repr(error),
            }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        cuda_check: Dict[str, Any] = {
            "ok": False,
            "required": True,
            "available": available,
            "build": torch.version.cuda,
            "compiled_architectures": (
                torch.cuda.get_arch_list() if torch.version.cuda is not None else []
            ),
            "device_count": torch.cuda.device_count(),
        }
        if available:
            try:
                selection = select_torch_device("auto")
                cuda_check.update(
                    {
                        "ok": selection.device.startswith("cuda"),
                        "kernel_verified": True,
                        "selection": selection.to_dict(),
                    }
                )
            except Exception as error:
                cuda_check.update({"kernel_verified": False, "error": repr(error)})
        checks["cuda"] = cuda_check
    except Exception as error:
        checks["cuda"] = {"ok": False, "required": True, "error": repr(error)}
    if config.paths.hm3d_root is not None and config.paths.hm3d_root.exists():
        for split in ("train", "val", "minival"):
            scenes = discover_scenes(config.paths.hm3d_root, split)
            checks[f"scenes:{split}"] = {
                "ok": bool(scenes),
                "annotated": len(scenes),
                "complete": sum(scene.complete for scene in scenes),
            }
        minival_scenes = discover_scenes(config.paths.hm3d_root, "minival")
        if minival_scenes and config.paths.scene_dataset_config is not None:
            probe_code = (
                "import habitat_sim,sys;"
                "b=habitat_sim.SimulatorConfiguration();"
                "b.scene_id=sys.argv[1];"
                "b.scene_dataset_config_file=sys.argv[2];"
                "s=habitat_sim.CameraSensorSpec();"
                "s.uuid='rgb';"
                "s.sensor_type=habitat_sim.SensorType.COLOR;"
                "s.resolution=[16,16];"
                "a=habitat_sim.AgentConfiguration();"
                "a.sensor_specifications=[s];"
                "sim=habitat_sim.Simulator(habitat_sim.Configuration(b,[a]));"
                "sim.get_sensor_observations();"
                "sim.close(destroy=True)"
            )
            try:
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        probe_code,
                        str(minival_scenes[0].rgb_mesh),
                        str(config.paths.scene_dataset_config),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                checks["headless_renderer"] = {
                    "ok": probe.returncode == 0,
                    "required": True,
                    "returncode": probe.returncode,
                    "stderr_tail": probe.stderr[-2000:],
                }
            except subprocess.TimeoutExpired as error:
                checks["headless_renderer"] = {
                    "ok": False,
                    "required": True,
                    "error": f"renderer probe timed out after {error.timeout} seconds",
                }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        checks["nvidia-smi"] = {
            "ok": result.returncode == 0,
            "required": True,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    else:
        checks["nvidia-smi"] = {
            "ok": False,
            "required": True,
            "error": "nvidia-smi executable was not found",
        }
    required_failures = [
        name
        for name, result in checks.items()
        if result.get("required", False) and not result.get("ok", False)
    ]
    return {
        "ok": not required_failures,
        "required_failures": required_failures,
        "python": sys.version,
        "provenance": collect_provenance(config.paths.habitat_lab_root),
        "checks": checks,
    }
