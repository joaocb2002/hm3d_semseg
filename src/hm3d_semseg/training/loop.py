"""Inspectable single-process SegFormer fine-tuning baseline."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
from PIL import Image

from hm3d_semseg.camera.profile import CameraProfile, assert_camera_compatible
from hm3d_semseg.config.loader import save_resolved_config
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.dataset import OfflineSegmentationDataset, select_manifest_records
from hm3d_semseg.data.schema import ManifestRecord, load_manifest
from hm3d_semseg.data.validate import validate_dataset
from hm3d_semseg.diagnostics.qualitative import (
    append_qualitative_epoch,
    capture_model_qualitative,
    select_qualitative_records,
    selection_report,
)
from hm3d_semseg.evaluation.run import evaluate_model
from hm3d_semseg.models.segformer import (
    build_segformer,
    parameter_groups,
    segmentation_objective,
)
from hm3d_semseg.training.artifacts import (
    development_evaluation_root,
    development_evaluations_root,
    plots_root,
    provenance_root,
    qualitative_root,
    records_root,
    report_root,
)
from hm3d_semseg.training.checkpoint import (
    load_training_state,
    restore_random_state,
    save_checkpoint,
    update_checkpoint_progress,
)
from hm3d_semseg.training.diagnostics import evaluate_training_subset
from hm3d_semseg.training.progress import TrainingProgress
from hm3d_semseg.training.report import generate_training_report
from hm3d_semseg.training.reporting import summarize_training_metrics
from hm3d_semseg.training.run_directory import allocate_run_directory
from hm3d_semseg.types import NumpyArray
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.hashing import atomic_write_json, sha256_file
from hm3d_semseg.utils.provenance import collect_provenance


def _seed_everything(seed: int, *, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _configure_deterministic_algorithms(enabled: bool) -> Dict[str, Any]:
    """Configure strict PyTorch kernels before any CUDA runtime probe."""
    import torch

    if enabled:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {None, ":4096:8", ":16:8"}:
            raise ValueError(
                "Strict deterministic training requires CUBLAS_WORKSPACE_CONFIG "
                "to be unset, ':4096:8', or ':16:8'"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
    return {
        "strict_algorithms": enabled,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(
            getattr(getattr(torch.backends, "cudnn", None), "deterministic", False)
        ),
        "cudnn_benchmark": bool(
            getattr(getattr(torch.backends, "cudnn", None), "benchmark", False)
        ),
    }


def _resolve_optimizer_steps(
    *,
    epochs: int,
    steps_per_epoch: int,
    maximum: Optional[int],
) -> int:
    """Resolve the epoch cap and optional iteration cap to one total."""
    epoch_steps = max(1, epochs * steps_per_epoch)
    return min(epoch_steps, maximum) if maximum is not None else epoch_steps


def _learning_rate_scale(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_start_factor: float,
    schedule: str,
    polynomial_power: float,
    legacy_warmup_offset: bool = False,
) -> float:
    """Return the official-style linear-warmup then decay multiplier."""
    if warmup_steps and step < warmup_steps:
        if legacy_warmup_offset:
            return float(step + 1) / warmup_steps
        if warmup_steps == 1:
            return 1.0
        fraction = step / (warmup_steps - 1)
        return warmup_start_factor + (1.0 - warmup_start_factor) * fraction
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "polynomial":
        return (1.0 - progress) ** polynomial_power
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported learning-rate schedule: {schedule}")


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scene_ids(dataset_root: Path) -> Set[str]:
    return {record.scene_id for record in load_manifest(dataset_root / "manifest.jsonl")}


def _resolve_class_weights(
    config: ProjectConfig, train_validation: Dict[str, Any], run: Path
) -> Optional[NumpyArray]:
    if config.training.class_weights is not None:
        weights = np.load(config.training.class_weights, allow_pickle=False)
        source = config.training.class_weights
        policy = "explicit"
    elif config.training.class_weighting == "inverse_sqrt":
        counts = np.asarray(train_validation["class_counts"], dtype=np.float64)
        supported = counts > 0
        weights = np.zeros(41, dtype=np.float32)
        weights[supported] = np.sqrt(
            counts[supported].sum() / (supported.sum() * counts[supported])
        )
        weights[supported] /= weights[supported].mean()
        weights = np.minimum(weights, config.training.class_weight_cap)
        source = provenance_root(run) / "class_weights.npy"
        with source.open("wb") as handle:
            np.save(handle, weights, allow_pickle=False)
        policy = "inverse_sqrt"
    else:
        return None
    if weights.shape != (41,) or not np.all(np.isfinite(weights)):
        raise ValueError("Class weights must contain exactly 41 finite values")
    atomic_write_json(
        provenance_root(run) / "class_weights.json",
        {
            "policy": policy,
            "source": str(source.resolve()),
            "sha256": sha256_file(source),
            "cap": config.training.class_weight_cap,
            "weights": weights.astype(float).tolist(),
            "training_manifest_only": True,
        },
    )
    return cast(NumpyArray, weights.astype(np.float32))


def _prepare_run_directories(run: Path) -> None:
    """Create the stable training-artifact directory hierarchy."""
    for directory in (
        run / "checkpoints",
        run / "tensorboard",
        records_root(run),
        provenance_root(run),
        qualitative_root(run) / "train",
        qualitative_root(run) / "development",
        development_evaluations_root(run),
        run / "diagnostics" / "train_subset",
        report_root(run) / "tables",
        plots_root(run) / "overview",
        plots_root(run) / "segmentation",
        plots_root(run) / "classes_and_scenes",
        plots_root(run) / "probability",
        plots_root(run) / "optimization",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _add_tensorboard_image(writer: Any, tag: str, path: Path, epoch: int) -> None:
    """Log a saved contact sheet without introducing a torchvision dependency."""
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    writer.add_image(tag, image, epoch, dataformats="HWC")


def _configure_tensorboard_layout(writer: Any) -> None:
    """Group the most useful curves into stable TensorBoard dashboards."""
    writer.add_custom_scalars(
        {
            "Generalization": {
                "cross_entropy": [
                    "Multiline",
                    ["train/epoch_loss", "development/loss"],
                ],
                "segmentation": [
                    "Multiline",
                    [
                        "development/known_class_miou",
                        "development/objectnav_six_miou",
                        "development/scene_macro_mean_miou",
                    ],
                ],
                "probability_quality": [
                    "Multiline",
                    [
                        "development/nll",
                        "development/multiclass_brier",
                        "development/ece",
                    ],
                ],
                "per_class_iou": [
                    "Multiline",
                    ["development/per_class_iou/.*"],
                ],
                "per_scene_miou": [
                    "Multiline",
                    ["development/per_scene_miou/.*"],
                ],
            },
            "Optimization": {
                "learning_rates": [
                    "Multiline",
                    [
                        "train/learning_rate_pretrained_decay",
                        "train/learning_rate_decode_head_decay",
                    ],
                ],
                "throughput": ["Multiline", ["train/samples_per_second"]],
                "health": [
                    "Multiline",
                    ["train/gradient_norm", "train/optimizer_step_skipped"],
                ],
            },
        }
    )


def train(config: ProjectConfig, *, show_progress: bool = True) -> Dict[str, Any]:
    """Train and resume a baseline while saving complete run provenance."""
    import torch
    from torch.utils.data import DataLoader

    progress = TrainingProgress(enabled=show_progress)
    if config.training.train_dataset is None:
        raise ValueError("training.train_dataset is required")
    if config.paths.runs_root is None:
        raise ValueError("paths.runs_root is required")
    if (
        config.training.early_stopping_patience is not None
        and config.training.development_dataset is None
    ):
        raise ValueError("Early stopping is allowed only with a development dataset")
    if (
        config.training.max_development_samples is not None
        and config.training.development_dataset is None
    ):
        raise ValueError(
            "training.max_development_samples requires a development dataset"
        )
    determinism = _configure_deterministic_algorithms(
        config.training.deterministic_algorithms
    )
    progress.message(
        "Strict deterministic PyTorch algorithms: "
        + ("enabled" if config.training.deterministic_algorithms else "disabled")
    )
    progress.message(f"Selecting training device (requested={config.training.device})...")
    device_selection = select_torch_device(config.training.device)
    device = device_selection.device
    using_cuda = device.startswith("cuda")
    progress.message(f"Selected device: {device}")
    manifest_records = load_manifest(config.training.train_dataset / "manifest.jsonl")
    selected_records = select_manifest_records(
        manifest_records,
        config.training.max_train_samples,
        strategy=config.training.sample_selection,
        seed=config.training.seed,
    )
    selected_sample_ids = [record.sample_id for record in selected_records]
    limited_training = config.training.max_train_samples is not None
    train_subset_evaluation_records: List[ManifestRecord] = []
    train_subset_evaluation_sample_ids: Optional[List[str]] = None
    if config.training.evaluate_train_subset:
        evaluation_limit = (
            config.training.train_subset_evaluation_samples
            if config.training.train_subset_evaluation_samples is not None
            else len(selected_records)
        )
        train_subset_evaluation_records = select_manifest_records(
            selected_records,
            evaluation_limit,
            strategy=config.training.sample_selection,
            seed=config.training.seed,
        )
        train_subset_evaluation_sample_ids = [
            record.sample_id for record in train_subset_evaluation_records
        ]
    if limited_training:
        progress.message(
            "Selected deterministic training subset: "
            f"{len(selected_records)} samples across "
            f"{len({record.scene_id for record in selected_records})} scenes "
            f"(strategy={config.training.sample_selection})"
        )
        progress.message(
            f"Validating selected training subset: {config.training.train_dataset}"
        )
    else:
        progress.message(f"Validating training dataset: {config.training.train_dataset}")
    train_validation = validate_dataset(
        config.training.train_dataset,
        sample_ids=selected_sample_ids if limited_training else None,
    )
    progress.message(
        f"Training {train_validation['validation_scope']} valid: "
        f"{train_validation['samples']} samples across {train_validation['scenes']} scenes "
        f"(manifest: {train_validation['manifest_samples']} samples, "
        f"{train_validation['manifest_scenes']} scenes)"
    )
    train_camera = CameraProfile.load(config.training.train_dataset / "camera_profile.yaml")
    if config.camera.profile is not None:
        assert_camera_compatible(
            CameraProfile.load(config.camera.profile),
            train_camera,
            config.camera.allow_mismatch,
        )
    development_validation = None
    development_records: List[ManifestRecord] = []
    development_sample_ids = None
    development_sample_selection = None
    limited_development = config.training.max_development_samples is not None
    if config.training.development_dataset is not None:
        development_sample_selection = (
            config.training.development_sample_selection
            if limited_development
            else "full_manifest"
        )
        development_manifest = load_manifest(
            config.training.development_dataset / "manifest.jsonl"
        )
        development_records = select_manifest_records(
            development_manifest,
            config.training.max_development_samples,
            strategy=config.training.development_sample_selection,
            seed=config.training.seed,
        )
        if limited_development:
            development_sample_ids = [record.sample_id for record in development_records]
            progress.message(
                "Selected deterministic development subset: "
                f"{len(development_records)} samples across "
                f"{len({record.scene_id for record in development_records})} scenes "
                f"(strategy={config.training.development_sample_selection})"
            )
            progress.message(
                "Validating selected development subset: "
                f"{config.training.development_dataset}"
            )
        else:
            progress.message(
                f"Validating development dataset: {config.training.development_dataset}"
            )
        development_validation = validate_dataset(
            config.training.development_dataset,
            sample_ids=development_sample_ids,
        )
        progress.message(
            f"Development {development_validation['validation_scope']} valid: "
            f"{development_validation['samples']} samples across "
            f"{development_validation['scenes']} scenes "
            f"(manifest: {development_validation['manifest_samples']} samples, "
            f"{development_validation['manifest_scenes']} scenes)"
        )
        development_camera = CameraProfile.load(
            config.training.development_dataset / "camera_profile.yaml"
        )
        assert_camera_compatible(train_camera, development_camera, config.camera.allow_mismatch)
        overlap = _scene_ids(config.training.train_dataset) & _scene_ids(
            config.training.development_dataset
        )
        if overlap:
            raise ValueError(
                "Train/development scene leakage: " + ", ".join(sorted(overlap)[:10])
            )
    _seed_everything(config.training.seed, seed_cuda=using_cuda)
    source_checkpoint = config.training.resume
    run = allocate_run_directory(
        config.paths.runs_root,
        config.training.run_name,
        resuming=source_checkpoint is not None,
    )
    if run.name != config.training.run_name:
        progress.message(
            f"Run '{config.training.run_name}' already exists; writing this fresh run "
            f"to '{run.name}'."
        )
    _prepare_run_directories(run)
    qualitative_train_records = select_qualitative_records(
        selected_records,
        config.training.qualitative_samples,
        seed=config.training.seed,
    )
    qualitative_development_records = (
        select_qualitative_records(
            development_records,
            config.training.qualitative_samples,
            seed=config.training.seed,
        )
        if development_records
        else []
    )
    qualitative_output_root = qualitative_root(run)
    atomic_write_json(
        qualitative_output_root / "selection.json",
        selection_report(
            qualitative_train_records,
            qualitative_development_records,
            seed=config.training.seed,
            requested_per_split=config.training.qualitative_samples,
        ),
    )
    save_resolved_config(config, provenance_root(run) / "resolved_config.yaml")
    provenance = collect_provenance(config.paths.habitat_lab_root)
    provenance.update(
        {
            "seed": config.training.seed,
            "determinism": determinism,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "device_selection": device_selection.to_dict(),
            "train_dataset_validation": train_validation,
            "development_dataset_validation": development_validation,
            "requested_run_name": config.training.run_name,
            "allocated_run": str(run),
            "training_sample_selection": config.training.sample_selection,
            "training_sample_ids": selected_sample_ids if limited_training else None,
            "train_subset_evaluation_sample_ids": (
                train_subset_evaluation_sample_ids
            ),
            "development_sample_selection": development_sample_selection,
            "development_sample_ids": development_sample_ids,
            "qualitative_train_sample_ids": [
                record.sample_id for record in qualitative_train_records
            ],
            "qualitative_development_sample_ids": [
                record.sample_id for record in qualitative_development_records
            ],
        }
    )
    atomic_write_json(provenance_root(run) / "provenance.json", provenance)

    progress.message(
        f"Loading model: {config.model.model_id}@{config.model.revision or 'unresolved'}"
    )
    model = build_segformer(
        config.model,
        checkpoint=source_checkpoint,
        cache_dir=config.paths.cache_root,
    ).to(device)
    head_learning_rate = (
        config.training.head_learning_rate
        if config.training.head_learning_rate is not None
        else config.training.classifier_learning_rate
    )
    groups = parameter_groups(
        model,
        config.training.encoder_learning_rate,
        head_learning_rate,
        config.training.weight_decay,
        entire_decode_head=config.training.head_learning_rate is not None,
        exclude_one_dimensional_from_decay=(
            config.training.head_learning_rate is not None
        ),
    )
    optimizer = torch.optim.AdamW(groups)
    dataset = OfflineSegmentationDataset(
        config.training.train_dataset,
        augment=True,
        augmentation=config.augmentation,
        seed=config.training.seed,
        sample_ids=selected_sample_ids if limited_training else None,
    )
    qualitative_train_dataset = OfflineSegmentationDataset(
        config.training.train_dataset,
        augment=False,
        sample_ids=[record.sample_id for record in qualitative_train_records],
    )
    loader: Any = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        pin_memory=using_cuda,
    )
    steps_per_epoch = max(
        1, math.ceil(len(loader) / config.training.gradient_accumulation_steps)
    )
    total_steps = _resolve_optimizer_steps(
        epochs=config.training.epochs,
        steps_per_epoch=steps_per_epoch,
        maximum=config.training.max_optimizer_steps,
    )
    schedule_steps = config.training.learning_rate_schedule_steps or total_steps

    warmup_steps = (
        config.training.warmup_steps
        if config.training.warmup_steps is not None
        else int(total_steps * config.training.warmup_fraction)
    )
    if warmup_steps >= schedule_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be smaller than the schedule horizon "
            f"({schedule_steps})"
        )

    def learning_rate_scale(step: int) -> float:
        return _learning_rate_scale(
            step,
            total_steps=schedule_steps,
            warmup_steps=warmup_steps,
            warmup_start_factor=config.training.warmup_start_factor,
            schedule=config.training.learning_rate_schedule,
            polynomial_power=config.training.polynomial_power,
            legacy_warmup_offset=config.training.warmup_steps is None,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_scale,
    )
    amp_enabled = config.training.amp and using_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_metric: Optional[float] = None
    best_development_loss: Optional[float] = None
    epochs_without_improvement = 0
    if source_checkpoint is not None and (source_checkpoint / "training_state.pt").is_file():
        state = load_training_state(source_checkpoint)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["step"])
        best_metric = state.get("primary_metric")
        best_development_loss = state.get("best_development_loss")
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        restore_random_state(state)

    resolved_weights = _resolve_class_weights(config, train_validation, run)
    class_weights = (
        torch.as_tensor(resolved_weights, dtype=torch.float32, device=device)
        if resolved_weights is not None
        else None
    )
    try:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard = SummaryWriter(log_dir=str(run / "tensorboard"))
        _configure_tensorboard_layout(tensorboard)
    except ImportError:
        tensorboard = None
    metrics_path = records_root(run) / "metrics.jsonl"
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer_groups = [
        {
            "name": str(group.get("group_name", f"group_{index}")),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "base_learning_rate": float(group.get("initial_lr", group["lr"])),
            "weight_decay": float(group["weight_decay"]),
        }
        for index, group in enumerate(groups)
    ]
    atomic_write_json(
        provenance_root(run) / "parameter_counts.json",
        {
            "trainable": trainable,
            "total": total,
            "optimizer_groups": optimizer_groups,
        },
    )
    progress.start(
        run=str(run),
        device=device,
        samples=len(dataset),
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        batches_per_epoch=len(loader),
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        completed_steps=global_step,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        amp=amp_enabled,
        trainable_parameters=trainable,
        total_parameters=total,
        encoder_learning_rate=config.training.encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=warmup_steps,
        learning_rate_schedule=config.training.learning_rate_schedule,
        learning_rate_schedule_steps=schedule_steps,
        cross_entropy_weight=config.training.loss.cross_entropy_weight,
        lovasz_weight=config.training.loss.lovasz_weight,
    )
    camera_path = config.training.train_dataset / "camera_profile.yaml"
    model_metadata = {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "num_labels": config.model.num_labels,
        "align_corners": config.model.align_corners,
        "reduce_labels": False,
        "training_loss": {
            "cross_entropy_weight": config.training.loss.cross_entropy_weight,
            "lovasz_weight": config.training.loss.lovasz_weight,
            "lovasz_include_unknown": config.training.loss.lovasz_include_unknown,
            "lovasz_resolution": config.training.loss.lovasz_resolution,
        },
    }
    epochs_completed = start_epoch
    for epoch in range(start_epoch, config.training.epochs):
        if global_step >= total_steps:
            break
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_cross_entropy = 0.0
        running_lovasz = 0.0
        batches = 0
        samples_since_step = 0
        if using_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimizer_step_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if global_step >= total_steps:
                break
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            samples_since_step += int(pixels.shape[0])
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                raw_logits = model(pixel_values=pixels).logits
                losses = segmentation_objective(
                    raw_logits,
                    labels,
                    cross_entropy_weight=(
                        config.training.loss.cross_entropy_weight
                    ),
                    lovasz_weight=config.training.loss.lovasz_weight,
                    lovasz_include_unknown=(
                        config.training.loss.lovasz_include_unknown
                    ),
                    lovasz_resolution=config.training.loss.lovasz_resolution,
                    align_corners=config.model.align_corners,
                    ignore_index=config.taxonomy.ignore_index,
                    class_weights=class_weights,
                )
                loss = losses.objective
                scaled_loss = loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}: {loss}"
                )
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
            running_cross_entropy += float(losses.cross_entropy.detach().cpu())
            running_lovasz += float(losses.lovasz.detach().cpu())
            batches += 1
            should_step = (
                batch_index + 1
            ) % config.training.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                loader
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                scale_before = float(scaler.get_scale()) if amp_enabled else 1.0
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale()) if amp_enabled else 1.0
                optimizer_step_skipped = amp_enabled and scale_after < scale_before
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if using_cuda:
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - optimizer_step_started
                samples_per_second = samples_since_step / max(step_seconds, 1e-12)
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(device)) if using_cuda else None
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "kind": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "objective_loss": float(loss.detach().cpu()),
                        "cross_entropy_loss": float(
                            losses.cross_entropy.detach().cpu()
                        ),
                        "lovasz_loss": float(losses.lovasz.detach().cpu()),
                        "cross_entropy_weight": (
                            config.training.loss.cross_entropy_weight
                        ),
                        "lovasz_weight": config.training.loss.lovasz_weight,
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "gradient_norm_finite": bool(
                            torch.isfinite(gradient_norm).detach().cpu()
                        ),
                        "optimizer_step_skipped": optimizer_step_skipped,
                        "learning_rates": [group["lr"] for group in optimizer.param_groups],
                        "samples": samples_since_step,
                        "step_seconds": step_seconds,
                        "samples_per_second": samples_per_second,
                        "gpu_peak_memory_bytes": peak_memory,
                    },
                )
                progress.step(
                    epoch=epoch,
                    loss=float(loss.detach().cpu()),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    samples_per_second=samples_per_second,
                )
                if tensorboard is not None:
                    tensorboard.add_scalar(
                        "train/loss", float(loss.detach().cpu()), global_step
                    )
                    tensorboard.add_scalar(
                        "train/objective_loss",
                        float(loss.detach().cpu()),
                        global_step,
                    )
                    tensorboard.add_scalar(
                        "train/cross_entropy_loss",
                        float(losses.cross_entropy.detach().cpu()),
                        global_step,
                    )
                    if config.training.loss.lovasz_weight > 0.0:
                        tensorboard.add_scalar(
                            "train/lovasz_loss",
                            float(losses.lovasz.detach().cpu()),
                            global_step,
                        )
                    tensorboard.add_scalar(
                        "train/gradient_norm",
                        float(gradient_norm.detach().cpu()),
                        global_step,
                    )
                    tensorboard.add_scalar(
                        "train/optimizer_step_skipped",
                        int(optimizer_step_skipped),
                        global_step,
                    )
                    for group_index, group in enumerate(optimizer.param_groups):
                        group_name = str(group.get("group_name", f"group_{group_index}"))
                        tensorboard.add_scalar(
                            f"train/learning_rate_{group_name}",
                            group["lr"],
                            global_step,
                        )
                    tensorboard.add_scalar(
                        "train/samples_per_second", samples_per_second, global_step
                    )
                    if peak_memory is not None:
                        tensorboard.add_scalar(
                            "train/gpu_peak_memory_bytes", peak_memory, global_step
                        )
                samples_since_step = 0
                if using_cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                optimizer_step_started = time.perf_counter()
        epoch_loss = running_loss / max(1, batches)
        epoch_cross_entropy = running_cross_entropy / max(1, batches)
        epoch_lovasz = running_lovasz / max(1, batches)
        _append_jsonl(
            metrics_path,
            {
                "kind": "train_epoch",
                "epoch": epoch,
                "loss": epoch_loss,
                "objective_loss": epoch_loss,
                "cross_entropy_loss": epoch_cross_entropy,
                "lovasz_loss": epoch_lovasz,
                "cross_entropy_weight": (
                    config.training.loss.cross_entropy_weight
                ),
                "lovasz_weight": config.training.loss.lovasz_weight,
            },
        )
        if tensorboard is not None:
            tensorboard.add_scalar("train/epoch_loss", epoch_loss, epoch)
            tensorboard.add_scalar(
                "train/epoch_objective_loss", epoch_loss, epoch
            )
            tensorboard.add_scalar(
                "train/epoch_cross_entropy_loss", epoch_cross_entropy, epoch
            )
            if config.training.loss.lovasz_weight > 0.0:
                tensorboard.add_scalar(
                    "train/epoch_lovasz_loss", epoch_lovasz, epoch
                )
        capture_qualitative = (
            epoch == start_epoch
            or (epoch + 1) % config.training.qualitative_every_epochs == 0
            or epoch + 1 == config.training.epochs
            or global_step >= total_steps
        )
        if capture_qualitative:
            progress.phase(epoch=epoch, name="training qualitative")
            train_qualitative = capture_model_qualitative(
                model,
                qualitative_train_dataset,
                qualitative_output_root / "train",
                epoch=epoch,
                device=device,
                align_corners=config.model.align_corners,
                ignore_index=config.taxonomy.ignore_index,
            )
            append_qualitative_epoch(
                qualitative_output_root / "train", train_qualitative
            )
            if tensorboard is not None:
                _add_tensorboard_image(
                    tensorboard,
                    "qualitative/train_fixed_set",
                    qualitative_output_root
                    / "train"
                    / train_qualitative["contact_sheet"],
                    epoch,
                )
        progress.phase(epoch=epoch, name="checkpoint")
        save_checkpoint(
            run / "checkpoints" / "last",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            primary_metric=best_metric,
            camera_profile_path=camera_path,
            epochs_without_improvement=epochs_without_improvement,
            best_development_loss=best_development_loss,
            model_metadata=model_metadata,
        )
        primary_metric = -epoch_loss
        if config.training.development_dataset is not None:
            progress.phase(epoch=epoch, name="development")
            evaluation = evaluate_model(
                run / "checkpoints" / "last",
                config.training.development_dataset,
                development_evaluation_root(run, epoch),
                config,
                device=device,
                sample_ids=development_sample_ids,
                qualitative_sample_ids=(
                    [record.sample_id for record in qualitative_development_records]
                    if capture_qualitative
                    else None
                ),
                qualitative_output=(
                    qualitative_output_root / "development"
                    if capture_qualitative
                    else None
                ),
                qualitative_epoch=epoch if capture_qualitative else None,
            )
            candidate = evaluation["global"]["known_class_miou"]
            development_loss = evaluation["mean_cross_entropy_loss"]
            primary_metric = float(candidate) if candidate is not None else -epoch_loss
            if (
                config.training.save_min_development_loss_checkpoint
                and development_loss is not None
                and (
                    best_development_loss is None
                    or float(development_loss) < best_development_loss
                )
            ):
                best_development_loss = float(development_loss)
                save_checkpoint(
                    run / "checkpoints" / "min_development_loss",
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    primary_metric=-best_development_loss,
                    camera_profile_path=camera_path,
                    epochs_without_improvement=epochs_without_improvement,
                    best_development_loss=best_development_loss,
                    model_metadata={
                        **model_metadata,
                        "selection_metric": "development_cross_entropy",
                        "selection_value": best_development_loss,
                    },
                )
            _append_jsonl(
                metrics_path,
                {
                    "kind": "development_epoch",
                    "epoch": epoch,
                    "loss": development_loss,
                    "known_class_miou": candidate,
                    "overall_pixel_accuracy": evaluation["global"]["overall_pixel_accuracy"],
                    "objectnav_six_miou": evaluation["global"]["objectnav_six_miou"],
                    "scene_macro_mean_miou": evaluation["scene_macro"]["mean"],
                    "nll": evaluation["probability_quality"]["nll"],
                    "ece": evaluation["probability_quality"]["ece"],
                    "multiclass_brier": evaluation["probability_quality"]["multiclass_brier"],
                },
            )
            if tensorboard is not None and development_loss is not None:
                tensorboard.add_scalar("development/loss", development_loss, epoch)
            if tensorboard is not None and candidate is not None:
                tensorboard.add_scalar("development/known_class_miou", candidate, epoch)
            if tensorboard is not None:
                for name, value in (
                    ("overall_pixel_accuracy", evaluation["global"]["overall_pixel_accuracy"]),
                    ("objectnav_six_miou", evaluation["global"]["objectnav_six_miou"]),
                    ("scene_macro_mean_miou", evaluation["scene_macro"]["mean"]),
                    ("nll", evaluation["probability_quality"]["nll"]),
                    ("ece", evaluation["probability_quality"]["ece"]),
                    ("multiclass_brier", evaluation["probability_quality"]["multiclass_brier"]),
                ):
                    if value is not None:
                        tensorboard.add_scalar(f"development/{name}", value, epoch)
                for class_metrics in evaluation["global"]["per_class"]:
                    if class_metrics["id"] > 0 and class_metrics["iou"] is not None:
                        tensorboard.add_scalar(
                            "development/per_class_iou/"
                            + str(class_metrics["name"]),
                            class_metrics["iou"],
                            epoch,
                        )
                for scene_id, scene_miou in evaluation["scene_macro"][
                    "per_scene_known_class_miou"
                ].items():
                    if scene_miou is not None:
                        tensorboard.add_scalar(
                            f"development/per_scene_miou/{scene_id}",
                            scene_miou,
                            epoch,
                        )
                qualitative = evaluation.get("qualitative")
                if qualitative is not None:
                    _add_tensorboard_image(
                        tensorboard,
                        "qualitative/development_fixed_set",
                        qualitative_output_root
                        / "development"
                        / qualitative["contact_sheet"],
                        epoch,
                    )
        improved = best_metric is None or primary_metric > best_metric
        if improved:
            best_metric = primary_metric
            epochs_without_improvement = 0
            save_checkpoint(
                run / "checkpoints" / "best",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                primary_metric=best_metric,
                camera_profile_path=camera_path,
                epochs_without_improvement=epochs_without_improvement,
                best_development_loss=best_development_loss,
                model_metadata=model_metadata,
            )
        else:
            epochs_without_improvement += 1
        update_checkpoint_progress(
            run / "checkpoints" / "last",
            primary_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            best_development_loss=best_development_loss,
        )
        epochs_completed = epoch + 1
        if (
            config.training.early_stopping_patience is not None
            and epochs_without_improvement >= config.training.early_stopping_patience
        ):
            _append_jsonl(
                metrics_path,
                {
                    "kind": "early_stopping",
                    "epoch": epoch,
                    "patience": config.training.early_stopping_patience,
                },
            )
            break
        if global_step >= total_steps:
            break
    progress.close()
    subset_evaluation_summary = None
    if config.training.evaluate_train_subset:
        diagnostic_checkpoint = run / "checkpoints" / "best"
        progress.message(
            "Evaluating generalization gap on a deterministic training subset "
            f"using {diagnostic_checkpoint}: "
            f"{len(train_subset_evaluation_records)} samples across "
            f"{len({record.scene_id for record in train_subset_evaluation_records})} "
            "scenes"
        )
        diagnostic_dataset = OfflineSegmentationDataset(
            config.training.train_dataset,
            augment=False,
            sample_ids=train_subset_evaluation_sample_ids,
        )
        diagnostic_model = build_segformer(
            config.model,
            checkpoint=diagnostic_checkpoint,
            cache_dir=config.paths.cache_root,
        ).to(device)
        diagnostic_report = evaluate_training_subset(
            diagnostic_model,
            diagnostic_dataset,
            run / "diagnostics" / "train_subset",
            config.model,
            checkpoint=diagnostic_checkpoint,
            device=device,
            ignore_index=config.taxonomy.ignore_index,
        )
        subset_evaluation_summary = {
            "report": str(run / "diagnostics" / "train_subset" / "summary.json"),
            "mean_cross_entropy_loss": diagnostic_report["mean_cross_entropy_loss"],
            "overall_pixel_accuracy": diagnostic_report["global"][
                "overall_pixel_accuracy"
            ],
            "known_class_miou": diagnostic_report["global"]["known_class_miou"],
        }
        if tensorboard is not None:
            if subset_evaluation_summary["overall_pixel_accuracy"] is not None:
                tensorboard.add_scalar(
                    "diagnostic/train_subset_pixel_accuracy",
                    subset_evaluation_summary["overall_pixel_accuracy"],
                    global_step,
                )
            if subset_evaluation_summary["known_class_miou"] is not None:
                tensorboard.add_scalar(
                    "diagnostic/train_subset_known_class_miou",
                    subset_evaluation_summary["known_class_miou"],
                    global_step,
                )
    summary = {
        "run": str(run),
        "requested_run_name": config.training.run_name,
        "epochs_completed": epochs_completed,
        "global_steps": global_step,
        "best_primary_metric": best_metric,
        "device": device,
        "amp": amp_enabled,
        "determinism": determinism,
        "class_weighting": config.training.class_weighting,
        "training_loss": {
            "cross_entropy_weight": config.training.loss.cross_entropy_weight,
            "lovasz_weight": config.training.loss.lovasz_weight,
            "lovasz_include_unknown": config.training.loss.lovasz_include_unknown,
            "lovasz_resolution": config.training.loss.lovasz_resolution,
        },
        "train_samples": len(dataset),
        "train_scenes": len({record.scene_id for record in dataset.records}),
        "training_sample_selection": config.training.sample_selection,
        "training_sample_ids": selected_sample_ids if limited_training else None,
        "training_validation_scope": train_validation["validation_scope"],
        "development_samples": (
            development_validation["samples"]
            if development_validation is not None
            else None
        ),
        "development_scenes": (
            development_validation["scenes"]
            if development_validation is not None
            else None
        ),
        "development_validation_scope": (
            development_validation["validation_scope"]
            if development_validation is not None
            else None
        ),
        "development_sample_selection": development_sample_selection,
        "development_sample_ids": development_sample_ids,
        "qualitative_samples_per_split": config.training.qualitative_samples,
        "qualitative_every_epochs": config.training.qualitative_every_epochs,
        "qualitative_train_sample_ids": [
            record.sample_id for record in qualitative_train_records
        ],
        "qualitative_development_sample_ids": [
            record.sample_id for record in qualitative_development_records
        ],
        "train_subset_evaluation": subset_evaluation_summary,
        "train_subset_evaluation_samples": (
            len(train_subset_evaluation_records)
            if config.training.evaluate_train_subset
            else None
        ),
        "train_subset_evaluation_sample_ids": train_subset_evaluation_sample_ids,
        "save_min_development_loss_checkpoint": (
            config.training.save_min_development_loss_checkpoint
        ),
        "best_development_loss": best_development_loss,
        "batch_size": config.training.batch_size,
        "batches_per_epoch": len(loader),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": total_steps,
        "max_optimizer_steps": config.training.max_optimizer_steps,
        "learning_rate_schedule": config.training.learning_rate_schedule,
        "learning_rate_schedule_steps": schedule_steps,
        "polynomial_power": config.training.polynomial_power,
        "warmup_steps": warmup_steps,
        "warmup_start_factor": config.training.warmup_start_factor,
        "warmup_mode": (
            "explicit_linear"
            if config.training.warmup_steps is not None
            else "legacy_fraction"
        ),
        "parameter_counts": {"trainable": trainable, "total": total},
        "optimizer_groups": optimizer_groups,
    }
    if tensorboard is not None:
        tensorboard.flush()
        tensorboard.close()
    metrics_summary = summarize_training_metrics(metrics_path)
    metrics_summary["plots"] = []
    metrics_summary["tensorboard"] = (
        "tensorboard" if tensorboard is not None else None
    )
    metrics_summary["train_subset_evaluation"] = subset_evaluation_summary
    development_metrics = metrics_summary.get("development")
    if development_metrics is not None:
        best_development = development_metrics.get("best_known_class_miou")
        if best_development is not None:
            epoch = int(best_development["epoch"])
            evaluation = development_evaluation_root(run, epoch)
            development_metrics["best_evaluation_report"] = str(
                (evaluation / "summary.json").relative_to(run)
            )
            development_metrics["best_evaluation_plots"] = str(
                (evaluation / "plots").relative_to(run)
            )
    metrics_summary_path = records_root(run) / "metrics_summary.json"
    atomic_write_json(metrics_summary_path, metrics_summary)
    summary["metrics_summary"] = str(metrics_summary_path)
    summary["metric_plots"] = metrics_summary["plots"]
    run_summary_path = records_root(run) / "run_summary.json"
    atomic_write_json(run_summary_path, summary)
    progress.message("Building human-readable training report...")
    human_report = generate_training_report(run)
    summary["human_report"] = human_report["report"]
    summary["metric_plots"] = human_report["plot_paths"]
    metrics_summary["plots"] = human_report["plot_paths"]
    metrics_summary["human_report"] = human_report["report"]
    atomic_write_json(metrics_summary_path, metrics_summary)
    atomic_write_json(run_summary_path, summary)
    return summary
