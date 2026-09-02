"""One consistent ``hm3d-semseg`` command surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from hm3d_semseg.audit.doctor import doctor as run_doctor
from hm3d_semseg.audit.taxonomy import audit_taxonomy as run_taxonomy_audit
from hm3d_semseg.calibration.temperature import fit_temperature
from hm3d_semseg.camera.profile import camera_differences, official_2023_profile
from hm3d_semseg.camera.resolve import resolve_camera_profile
from hm3d_semseg.config.loader import load_config
from hm3d_semseg.data.generate import generate_dataset as run_generation
from hm3d_semseg.data.progress import DatasetGenerationProgress
from hm3d_semseg.data.schema import load_manifest
from hm3d_semseg.data.splits import make_calibration_split, make_development_split
from hm3d_semseg.data.validate import validate_dataset as run_dataset_validation
from hm3d_semseg.diagnostics.qualitative import (
    select_qualitative_records,
    selection_report,
)
from hm3d_semseg.diagnostics.smoke import run_smoke_test
from hm3d_semseg.evaluation.benchmark import benchmark_inference
from hm3d_semseg.evaluation.reporting import generate_evaluation_report
from hm3d_semseg.evaluation.run import evaluate_model
from hm3d_semseg.exceptions import HM3DSemsegError
from hm3d_semseg.inference.api import SemanticSegmenter
from hm3d_semseg.installation.training_env import install_training_environment
from hm3d_semseg.models.segformer import download_model as run_model_download
from hm3d_semseg.scenes.inspect import inspect_scene as run_scene_inspection
from hm3d_semseg.training.loop import train as run_training
from hm3d_semseg.training.report import compare_training_runs, generate_training_report
from hm3d_semseg.utils.hashing import atomic_write_json

app = typer.Typer(
    name="hm3d-semseg",
    help="HM3D semantic rendering and 41-class SegFormer workflows.",
    no_args_is_help=True,
)


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command("install-training-env")
def install_training_env(
    project_root: Path = typer.Option(
        Path.cwd(), "--project-root", exists=True, file_okay=False
    ),
    profile: str = typer.Option("auto", "--profile"),
    apply: bool = typer.Option(False, "--apply"),
    with_dev: bool = typer.Option(True, "--with-dev/--without-dev"),
    force_torch: bool = typer.Option(False, "--force-torch"),
    run_tests: bool = typer.Option(False, "--run-tests"),
    allow_unsupported_host: bool = typer.Option(False, "--allow-unsupported-host"),
) -> None:
    """Plan or apply a host-matched CPU/CUDA training environment."""
    _print(
        install_training_environment(
            project_root,
            profile_name=profile,
            apply=apply,
            with_dev=with_dev,
            force_torch=force_torch,
            run_tests=run_tests,
            allow_unsupported_host=allow_unsupported_host,
        )
    )


@app.command()
def doctor(
    local_config: Path = typer.Option(..., "--local-config", exists=True, dir_okay=False),
) -> None:
    """Diagnose dependencies, data paths, annotated scenes, CUDA, and driver state."""
    report = run_doctor(load_config(local_config=local_config))
    _print(report)
    if not report["ok"]:
        raise typer.Exit(1)


@app.command("resolve-camera")
def resolve_camera(
    output: Path = typer.Option(..., "--output", dir_okay=False),
    local_config: Optional[Path] = typer.Option(
        None, "--local-config", exists=True, dir_okay=False
    ),
    objectnav_config: Optional[Path] = typer.Option(
        None, "--objectnav-config", exists=True, dir_okay=False
    ),
    raw_yaml_fallback: bool = typer.Option(False, "--raw-yaml-fallback"),
) -> None:
    """Compose and freeze the actual ObjectNav camera contract."""
    config = load_config(local_config=local_config)
    source = objectnav_config or config.paths.objectnav_config
    if source is None:
        raise typer.BadParameter(
            "Provide --objectnav-config or paths.objectnav_config in local config"
        )
    profile = resolve_camera_profile(source, config.paths.habitat_lab_root, raw_yaml_fallback)
    profile.save(output)
    official = official_2023_profile()
    _print(
        {
            "output": str(output.resolve()),
            "profile": profile.to_dict(),
            "official_2023_comparison": camera_differences(official, profile),
        }
    )


@app.command("inspect-scene")
def inspect_scene(
    local_config: Path = typer.Option(..., "--local-config", exists=True),
    split: str = typer.Option("minival", "--split"),
    scene_id: str = typer.Option(..., "--scene-id"),
    num_views: int = typer.Option(4, "--num-views", min=1),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Render aligned RGB, semantic IDs, mapped masks, depth, and overlays."""
    _print(
        run_scene_inspection(
            load_config(local_config=local_config),
            split,
            scene_id,
            num_views,
            output,
        )
    )


@app.command("audit-taxonomy")
def audit_taxonomy(
    local_config: Path = typer.Option(..., "--local-config", exists=True),
    split: str = typer.Option(..., "--split"),
    output: Path = typer.Option(..., "--output"),
    rendered_dataset: Optional[Path] = typer.Option(None, "--rendered-dataset"),
) -> None:
    """Audit raw labels, mapping policy, class support, and optional pixels."""
    _print(
        run_taxonomy_audit(
            load_config(local_config=local_config),
            split,
            output,
            rendered_dataset,
        )
    )


@app.command("generate-dataset")
def generate_dataset(
    config: Path = typer.Option(..., "--config", exists=True),
    local_config: Path = typer.Option(..., "--local-config", exists=True),
    split_list: Optional[Path] = typer.Option(None, "--split-list", exists=True),
    max_scenes: Optional[int] = typer.Option(None, "--max-scenes", min=1),
    max_samples_per_scene: Optional[int] = typer.Option(None, "--max-samples-per-scene", min=1),
    official_split: Optional[str] = typer.Option(None, "--official-split"),
    dataset_name: Optional[str] = typer.Option(None, "--dataset-name"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    show_progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Generate or resume a deterministic offline dataset, grouped by scene."""
    dataset_overrides: Dict[str, Any] = {}
    if official_split is not None:
        dataset_overrides["split"] = official_split
        dataset_overrides["name"] = f"official-{official_split}-v1"
    if dataset_name is not None:
        dataset_overrides["name"] = dataset_name
    resolved = load_config(
        command_config=config,
        local_config=local_config,
        cli_overrides={"dataset": dataset_overrides} if dataset_overrides else None,
    )
    plan = run_generation(
        resolved,
        split_list=split_list,
        max_scenes=max_scenes,
        max_samples_per_scene=max_samples_per_scene,
        validation_only=True,
    )
    if dry_run:
        _print(plan)
        return
    _print({"generation_plan": plan})
    with DatasetGenerationProgress(enabled=show_progress) as progress:
        result = run_generation(
            resolved,
            split_list=split_list,
            max_scenes=max_scenes,
            max_samples_per_scene=max_samples_per_scene,
            validation_only=False,
            progress=progress,
        )
    _print(result)


@app.command("validate-dataset")
def validate_dataset(
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
) -> None:
    """Check every manifest record, image pair, class target, and scene split."""
    _print(run_dataset_validation(dataset, artifact_output=dataset / "validation"))


@app.command("download-model")
def download_model(
    model_id: str = typer.Option("nvidia/segformer-b2-finetuned-ade-512-512", "--model-id"),
    local_config: Path = typer.Option(..., "--local-config", exists=True),
    revision: Optional[str] = typer.Option(None, "--revision"),
) -> None:
    """Explicitly download, resolve, and record a pinned pretrained snapshot."""
    config = load_config(local_config=local_config)
    if config.paths.cache_root is None:
        raise typer.BadParameter("paths.cache_root is required")
    provenance = run_model_download(model_id, config.paths.cache_root, revision)
    atomic_write_json(config.paths.cache_root / "model_download.json", provenance)
    _print(provenance)


@app.command("make-dev-split")
def make_dev_split(
    audit: Path = typer.Option(..., "--audit", exists=True),
    output: Path = typer.Option(..., "--output"),
    local_config: Optional[Path] = typer.Option(None, "--local-config", exists=True),
    development_scenes: int = typer.Option(15, "--development-scenes", min=1),
) -> None:
    """Freeze deterministic scene-disjoint fit and development lists."""
    config = load_config(local_config=local_config)
    _print(make_development_split(audit, output, config.sampling.seed, development_scenes))


@app.command("make-calibration-split")
def make_calibration_scene_split(
    audit: Path = typer.Option(..., "--audit", exists=True),
    output: Path = typer.Option(..., "--output"),
    local_config: Optional[Path] = typer.Option(None, "--local-config", exists=True),
    fit_scenes: int = typer.Option(12, "--fit-scenes", min=1),
) -> None:
    """Freeze disjoint temperature-fit and calibration-evaluation scene lists."""
    config = load_config(local_config=local_config)
    _print(make_calibration_split(audit, output, config.evaluation.bootstrap_seed, fit_scenes))


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True),
    local_config: Path = typer.Option(..., "--local-config", exists=True),
    show_progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Fine-tune SegFormer, with exact resume and best/last checkpoints."""
    _print(
        run_training(
            load_config(command_config=config, local_config=local_config),
            show_progress=show_progress,
        )
    )


@app.command("report-run")
def report_run(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
) -> None:
    """Build or refresh a static human-readable report from one run directory."""
    _print(generate_training_report(run))


@app.command("compare-runs")
def compare_runs(
    runs: List[Path] = typer.Option(..., "--run", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Compare held-out results from two or more protocol-compatible runs."""
    _print(compare_training_runs(runs, output))


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, file_okay=False),
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    local_config: Optional[Path] = typer.Option(None, "--local-config", exists=True),
    temperature: float = typer.Option(1.0, "--temperature", min=0.001),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    """Evaluate global, scene-macro, ObjectNav-six, and probability metrics."""
    resolved = load_config(command_config=config, local_config=local_config)
    qualitative_records = select_qualitative_records(
        load_manifest(dataset / "manifest.jsonl"),
        resolved.evaluation.qualitative_samples,
        seed=resolved.evaluation.bootstrap_seed,
    )
    qualitative_output = output / "qualitative"
    atomic_write_json(
        qualitative_output / "selection.json",
        selection_report(
            [],
            [],
            seed=resolved.evaluation.bootstrap_seed,
            requested_per_split=resolved.evaluation.qualitative_samples,
            evaluation_records=qualitative_records,
        ),
    )
    result = evaluate_model(
        checkpoint,
        dataset,
        output,
        resolved,
        temperature=temperature,
        device=device,
        qualitative_sample_ids=[record.sample_id for record in qualitative_records],
        qualitative_output=qualitative_output,
        qualitative_epoch=0,
    )
    result["human_report"] = generate_evaluation_report(output)["report"]
    _print(result)


@app.command()
def calibrate(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, file_okay=False),
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    local_config: Optional[Path] = typer.Option(None, "--local-config", exists=True),
    epochs: int = typer.Option(5, "--epochs", min=1),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    """Fit scalar temperature on dedicated scenes and preserve provenance."""
    resolved = load_config(command_config=config, local_config=local_config)
    _print(
        fit_temperature(
            checkpoint,
            dataset,
            output,
            resolved,
            epochs=epochs,
            device=device,
        )
    )


@app.command("benchmark-inference")
def benchmark_inference_command(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    iterations: int = typer.Option(100, "--iterations", min=1),
    warmup: int = typer.Option(20, "--warmup", min=0),
    device: Optional[str] = typer.Option(None, "--device"),
    float32: bool = typer.Option(False, "--float32"),
) -> None:
    """Measure native-camera batch-1 latency, FPS, size, and peak memory."""
    _print(
        benchmark_inference(
            checkpoint,
            output,
            iterations=iterations,
            warmup=warmup,
            device=device,
            half_precision=not float32,
        )
    )


@app.command()
def infer(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, file_okay=False),
    image: Path = typer.Option(..., "--image", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    save_probabilities: bool = typer.Option(False, "--save-probabilities"),
    device: Optional[str] = typer.Option(None, "--device"),
) -> None:
    """Save class IDs, overlay, confidence, entropy, and optional 41-way tensor."""
    segmenter = SemanticSegmenter.from_checkpoint(checkpoint, device=device)
    _print(segmenter.infer_file(image, output, save_probabilities=save_probabilities))


@app.command("smoke-test")
def smoke_test(
    local_config: Path = typer.Option(..., "--local-config", exists=True),
) -> None:
    """Render, train, evaluate, reload, and infer through a tiny diagnostic run."""
    config = load_config(local_config=local_config)
    report = run_doctor(config)
    if not report["ok"]:
        _print(report)
        raise typer.Exit(1)
    _print(run_smoke_test(config))


def main() -> None:
    """Console-script wrapper with concise expected-error handling."""
    try:
        app()
    except HM3DSemsegError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error


if __name__ == "__main__":
    main()
