"""Inspectable single-process SegFormer fine-tuning baseline."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from hm3d_semseg.camera.profile import CameraProfile, assert_camera_compatible
from hm3d_semseg.config.loader import save_resolved_config
from hm3d_semseg.config.schema import ProjectConfig
from hm3d_semseg.data.dataset import OfflineSegmentationDataset
from hm3d_semseg.data.schema import load_manifest
from hm3d_semseg.data.storage import load_mask
from hm3d_semseg.data.validate import validate_dataset
from hm3d_semseg.evaluation.run import evaluate_model
from hm3d_semseg.models.segformer import (
    build_segformer,
    parameter_groups,
    predict,
    segmentation_loss,
)
from hm3d_semseg.training.checkpoint import (
    load_training_state,
    restore_random_state,
    save_checkpoint,
    update_checkpoint_progress,
)
from hm3d_semseg.training.plots import save_training_plots
from hm3d_semseg.training.progress import TrainingProgress
from hm3d_semseg.utils.device import select_torch_device
from hm3d_semseg.utils.hashing import atomic_write_json, sha256_file
from hm3d_semseg.utils.provenance import collect_provenance
from hm3d_semseg.visualization.masks import overlay_mask


def _seed_everything(seed: int, *, seed_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scene_ids(dataset_root: Path) -> set:
    return {record.scene_id for record in load_manifest(dataset_root / "manifest.jsonl")}


def _resolve_class_weights(
    config: ProjectConfig, train_validation: Dict[str, Any], run: Path
) -> Optional[np.ndarray]:
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
        source = run / "class_weights.npy"
        with source.open("wb") as handle:
            np.save(handle, weights, allow_pickle=False)
        policy = "inverse_sqrt"
    else:
        return None
    if weights.shape != (41,) or not np.all(np.isfinite(weights)):
        raise ValueError("Class weights must contain exactly 41 finite values")
    atomic_write_json(
        run / "class_weights.json",
        {
            "policy": policy,
            "source": str(source.resolve()),
            "sha256": sha256_file(source),
            "cap": config.training.class_weight_cap,
            "weights": weights.astype(float).tolist(),
            "training_manifest_only": True,
        },
    )
    return weights.astype(np.float32)


def _save_qualitative(
    model: Any,
    dataset_root: Path,
    model_config: Any,
    output: Path,
    device: str,
) -> None:
    import torch

    fixed = OfflineSegmentationDataset(dataset_root, augment=False)
    if not fixed.records:
        return
    item = fixed[0]
    record = fixed.records[0]
    with Image.open(dataset_root / record.rgb) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    target = load_mask(dataset_root / record.mask)
    model.eval()
    with torch.inference_mode():
        result = predict(
            model,
            item["pixel_values"].unsqueeze(0).to(device),
            output_size=target.shape,
            align_corners=model_config.align_corners,
        )
    prediction = result.labels[0].to(torch.uint8).cpu().numpy()
    panel = np.concatenate(
        [rgb, overlay_mask(rgb, target), overlay_mask(rgb, prediction)], axis=1
    )
    Image.fromarray(panel, mode="RGB").save(output)


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
    progress.message(f"Selecting training device (requested={config.training.device})...")
    device_selection = select_torch_device(config.training.device)
    device = device_selection.device
    using_cuda = device.startswith("cuda")
    progress.message(f"Selected device: {device}")
    progress.message(f"Validating training dataset: {config.training.train_dataset}")
    train_validation = validate_dataset(config.training.train_dataset)
    progress.message(
        "Training dataset valid: "
        f"{train_validation['samples']} samples across {train_validation['scenes']} scenes"
    )
    train_camera = CameraProfile.load(config.training.train_dataset / "camera_profile.yaml")
    if config.camera.profile is not None:
        assert_camera_compatible(
            CameraProfile.load(config.camera.profile),
            train_camera,
            config.camera.allow_mismatch,
        )
    development_validation = None
    if config.training.development_dataset is not None:
        progress.message(
            f"Validating development dataset: {config.training.development_dataset}"
        )
        development_validation = validate_dataset(config.training.development_dataset)
        progress.message(
            "Development dataset valid: "
            f"{development_validation['samples']} samples across "
            f"{development_validation['scenes']} scenes"
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
    run = config.paths.runs_root / config.training.run_name
    run.mkdir(parents=True, exist_ok=True)
    for directory in (
        "checkpoints",
        "tensorboard",
        "plots",
        "qualitative",
    ):
        (run / directory).mkdir(exist_ok=True)
    save_resolved_config(config, run / "resolved_config.yaml")
    provenance = collect_provenance(config.paths.habitat_lab_root)
    provenance.update(
        {
            "seed": config.training.seed,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "device_selection": device_selection.to_dict(),
            "train_dataset_validation": train_validation,
            "development_dataset_validation": development_validation,
        }
    )
    atomic_write_json(run / "provenance.json", provenance)

    source_checkpoint = config.training.resume
    progress.message(
        f"Loading model: {config.model.model_id}@{config.model.revision or 'unresolved'}"
    )
    model = build_segformer(
        config.model,
        checkpoint=source_checkpoint,
        cache_dir=config.paths.cache_root,
    ).to(device)
    groups = parameter_groups(
        model,
        config.training.encoder_learning_rate,
        config.training.classifier_learning_rate,
        config.training.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups)
    dataset = OfflineSegmentationDataset(
        config.training.train_dataset,
        augment=True,
        augmentation=config.augmentation,
        seed=config.training.seed,
        max_samples=config.training.max_train_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        pin_memory=using_cuda,
    )
    steps_per_epoch = max(
        1, math.ceil(len(loader) / config.training.gradient_accumulation_steps)
    )
    total_steps = max(1, config.training.epochs * steps_per_epoch)

    warmup_steps = int(total_steps * config.training.warmup_fraction)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_scale,
    )
    amp_enabled = config.training.amp and using_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_metric: Optional[float] = None
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
    except ImportError:
        tensorboard = None
    metrics_path = run / "metrics.jsonl"
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    atomic_write_json(run / "parameter_counts.json", {"trainable": trainable, "total": total})
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
        classifier_learning_rate=config.training.classifier_learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=warmup_steps,
    )
    camera_path = config.training.train_dataset / "camera_profile.yaml"
    model_metadata = {
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "num_labels": config.model.num_labels,
        "align_corners": config.model.align_corners,
        "reduce_labels": False,
    }
    epochs_completed = start_epoch
    for epoch in range(start_epoch, config.training.epochs):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        samples_since_step = 0
        if using_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        optimizer_step_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            samples_since_step += int(pixels.shape[0])
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                raw_logits = model(pixel_values=pixels).logits
                loss = segmentation_loss(
                    raw_logits,
                    labels,
                    align_corners=config.model.align_corners,
                    ignore_index=config.taxonomy.ignore_index,
                    class_weights=class_weights,
                )
                scaled_loss = loss / config.training.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}: {loss}"
                )
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())
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
                scaler.step(optimizer)
                scaler.update()
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
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rates": [group["lr"] for group in optimizer.param_groups],
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
                        "train/gradient_norm",
                        float(gradient_norm.detach().cpu()),
                        global_step,
                    )
                    for group_index, group in enumerate(optimizer.param_groups):
                        tensorboard.add_scalar(
                            f"train/learning_rate_{group_index}",
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
        _append_jsonl(
            metrics_path,
            {"kind": "train_epoch", "epoch": epoch, "loss": epoch_loss},
        )
        if tensorboard is not None:
            tensorboard.add_scalar("train/epoch_loss", epoch_loss, epoch)
        _save_qualitative(
            model,
            config.training.train_dataset,
            config.model,
            run / "qualitative" / f"epoch_{epoch:03d}.png",
            device,
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
            model_metadata=model_metadata,
        )
        primary_metric = -epoch_loss
        if config.training.development_dataset is not None:
            progress.phase(epoch=epoch, name="development")
            evaluation = evaluate_model(
                run / "checkpoints" / "last",
                config.training.development_dataset,
                run / f"evaluation-epoch-{epoch:03d}",
                config,
                device=device,
            )
            candidate = evaluation["global"]["known_class_miou"]
            development_loss = evaluation["mean_cross_entropy_loss"]
            primary_metric = float(candidate) if candidate is not None else -epoch_loss
            _append_jsonl(
                metrics_path,
                {
                    "kind": "development_epoch",
                    "epoch": epoch,
                    "loss": development_loss,
                    "known_class_miou": candidate,
                },
            )
            if tensorboard is not None and development_loss is not None:
                tensorboard.add_scalar("development/loss", development_loss, epoch)
            if tensorboard is not None and candidate is not None:
                tensorboard.add_scalar("development/known_class_miou", candidate, epoch)
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
                model_metadata=model_metadata,
            )
        else:
            epochs_without_improvement += 1
        update_checkpoint_progress(
            run / "checkpoints" / "last",
            primary_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
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
    progress.close()
    summary = {
        "run": str(run),
        "epochs_completed": epochs_completed,
        "global_steps": global_step,
        "best_primary_metric": best_metric,
        "device": device,
        "amp": amp_enabled,
        "class_weighting": config.training.class_weighting,
        "train_samples": len(dataset),
        "batch_size": config.training.batch_size,
        "batches_per_epoch": len(loader),
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "parameter_counts": {"trainable": trainable, "total": total},
    }
    if tensorboard is not None:
        tensorboard.flush()
        tensorboard.close()
    save_training_plots(metrics_path, run / "plots")
    atomic_write_json(run / "summary.json", summary)
    return summary
