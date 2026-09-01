"""Hugging Face SegFormer-B2 adapted to 41 project classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hm3d_semseg.config.schema import ModelConfig
from hm3d_semseg.exceptions import ConfigurationError, OptionalDependencyError
from hm3d_semseg.taxonomy.constants import ID2LABEL, LABEL2ID, NUM_CLASSES


@dataclass
class SegmentationOutput:
    logits: Any
    probabilities: Any
    labels: Any
    confidence: Any
    entropy: Any


@dataclass
class SegmentationLosses:
    """Differentiable objective and its unscaled, human-readable components."""

    objective: Any
    cross_entropy: Any
    lovasz: Any


def _torch_imports() -> Tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise OptionalDependencyError(
            "PyTorch is required; install the train extra."
        ) from error
    return torch, functional


def _model_class() -> Any:
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as error:
        raise OptionalDependencyError(
            "Transformers is required; install the train extra."
        ) from error
    return SegformerForSemanticSegmentation


def build_segformer(
    config: ModelConfig,
    checkpoint: Optional[Path] = None,
    output_loading_info: bool = False,
    cache_dir: Optional[Path] = None,
) -> Any:
    """Load pretrained weights while allowing only the 150→41 classifier mismatch."""
    if (
        checkpoint is None
        and config.revision is None
        and not Path(config.model_id).expanduser().exists()
    ):
        raise ConfigurationError(
            "model.revision must be a pinned Hugging Face commit. Run "
            "`hm3d-semseg download-model`, then record its resolved_revision."
        )
    model_class = _model_class()
    source = str(checkpoint) if checkpoint is not None else config.model_id
    result = model_class.from_pretrained(
        source,
        revision=config.revision,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        local_files_only=config.local_files_only,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        output_loading_info=True,
    )
    model, loading = result
    mismatched = loading.get("mismatched_keys", [])
    unexpected_mismatches = [
        item
        for item in mismatched
        if "decode_head.classifier" not in str(item[0] if isinstance(item, tuple) else item)
    ]
    unexpected_missing = [
        key
        for key in loading.get("missing_keys", [])
        if "decode_head.classifier" not in str(key)
    ]
    unexpected_parameters = [
        key
        for key in loading.get("unexpected_keys", [])
        if "decode_head.classifier" not in str(key)
    ]
    if unexpected_mismatches or unexpected_missing or unexpected_parameters:
        raise RuntimeError(
            "Unexpected pretrained loading differences: "
            f"mismatched={unexpected_mismatches}, missing={unexpected_missing}, "
            f"unexpected={unexpected_parameters}"
        )
    if model.config.num_labels != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} labels, found {model.config.num_labels}")
    model.config.reduce_labels = False
    model.config.semantic_loss_ignore_index = 255
    model.config.hm3d_semseg_align_corners = config.align_corners
    if checkpoint is None:
        model.config.hm3d_semseg_source_model_id = config.model_id
        model.config.hm3d_semseg_source_revision = config.revision
    return (model, loading) if output_loading_info else model


def upsample_logits(logits: Any, output_size: Tuple[int, int], align_corners: bool) -> Any:
    _, functional = _torch_imports()
    return functional.interpolate(
        logits,
        size=output_size,
        mode="bilinear",
        align_corners=align_corners,
    )


def segmentation_loss(
    raw_logits: Any,
    targets: Any,
    *,
    align_corners: bool = False,
    ignore_index: int = 255,
    class_weights: Optional[Any] = None,
) -> Any:
    """Cross-entropy on raw upsampled logits; no softmax is applied."""
    _, functional = _torch_imports()
    logits = upsample_logits(raw_logits, targets.shape[-2:], align_corners)
    return functional.cross_entropy(
        logits, targets, weight=class_weights, ignore_index=ignore_index
    )


def lovasz_softmax_loss(
    raw_logits: Any,
    targets: Any,
    *,
    align_corners: bool = False,
    ignore_index: int = 255,
    include_unknown: bool = False,
    resolution: str = "native",
) -> Any:
    """Lovasz-Softmax over ground-truth-present classes in one minibatch.

    ``native`` evaluates the surrogate at the decoder-logit resolution and
    downsamples the discrete target with nearest interpolation. ``target``
    upsamples logits to the stored target size. Target 255 is always removed;
    class 0 remains a valid negative even when it is excluded from the macro
    set of positive classes.
    """
    torch, functional = _torch_imports()
    if resolution == "native":
        logits = raw_logits
        if tuple(targets.shape[-2:]) != tuple(raw_logits.shape[-2:]):
            labels = functional.interpolate(
                targets.unsqueeze(1).to(dtype=torch.float32),
                size=raw_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).to(dtype=targets.dtype)
        else:
            labels = targets
    elif resolution == "target":
        logits = upsample_logits(raw_logits, targets.shape[-2:], align_corners)
        labels = targets
    else:
        raise ValueError("Lovasz resolution must be 'native' or 'target'")

    probabilities = torch.softmax(logits.float(), dim=1)
    probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)
    valid = labels != ignore_index
    probabilities = probabilities[valid]
    labels = labels[valid]
    if labels.numel() == 0:
        return probabilities.sum() * 0.0

    losses = []
    class_ids = [
        int(value)
        for value in torch.unique(labels).tolist()
        if 0 <= int(value) < int(logits.shape[1])
        and (include_unknown or int(value) != 0)
    ]
    for class_id in class_ids:
        foreground = (labels == class_id).to(dtype=probabilities.dtype)
        errors = (foreground - probabilities[:, class_id]).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        gradient = _lovasz_gradient(foreground_sorted)
        losses.append(torch.dot(errors_sorted, gradient))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def _lovasz_gradient(sorted_foreground: Any) -> Any:
    """Discrete derivative of the Jaccard loss for sorted pixel errors."""
    foreground_count = sorted_foreground.sum()
    intersection = foreground_count - sorted_foreground.cumsum(0)
    union = foreground_count + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union
    if gradient.numel() > 1:
        torch, _ = _torch_imports()
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def segmentation_objective(
    raw_logits: Any,
    targets: Any,
    *,
    cross_entropy_weight: float = 1.0,
    lovasz_weight: float = 0.0,
    lovasz_include_unknown: bool = False,
    lovasz_resolution: str = "native",
    align_corners: bool = False,
    ignore_index: int = 255,
    class_weights: Optional[Any] = None,
) -> SegmentationLosses:
    """Return the configured loss and its raw CE/Lovasz components."""
    cross_entropy = segmentation_loss(
        raw_logits,
        targets,
        align_corners=align_corners,
        ignore_index=ignore_index,
        class_weights=class_weights,
    )
    if lovasz_weight > 0.0:
        lovasz = lovasz_softmax_loss(
            raw_logits,
            targets,
            align_corners=align_corners,
            ignore_index=ignore_index,
            include_unknown=lovasz_include_unknown,
            resolution=lovasz_resolution,
        )
    else:
        lovasz = cross_entropy.detach() * 0.0
    objective = (
        cross_entropy
        if cross_entropy_weight == 1.0 and lovasz_weight == 0.0
        else cross_entropy_weight * cross_entropy + lovasz_weight * lovasz
    )
    return SegmentationLosses(objective, cross_entropy, lovasz)


def predict(
    model: Any,
    pixel_values: Any,
    *,
    output_size: Tuple[int, int],
    align_corners: bool = False,
    temperature: float = 1.0,
) -> SegmentationOutput:
    """Return the categorical 41-way distribution and derived maps."""
    torch, _ = _torch_imports()
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    raw = model(pixel_values=pixel_values).logits
    logits = upsample_logits(raw, output_size, align_corners)
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidence, labels = probabilities.max(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return SegmentationOutput(logits, probabilities, labels, confidence, entropy)


def parameter_groups(
    model: Any,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    *,
    entire_decode_head: bool = False,
    exclude_one_dimensional_from_decay: bool = False,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, bool], List[Any]] = {
        ("pretrained", True): [],
        ("pretrained", False): [],
        ("decode_head", True): [],
        ("decode_head", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        high_rate = (
            name.startswith("decode_head.")
            if entire_decode_head
            else "decode_head.classifier" in name
        )
        family = "decode_head" if high_rate else "pretrained"
        use_decay = not exclude_one_dimensional_from_decay or (
            parameter.ndim > 1 and not name.endswith(".bias")
        )
        grouped[(family, use_decay)].append(parameter)
    result = []
    for family in ("pretrained", "decode_head"):
        learning_rate = (
            head_learning_rate
            if family == "decode_head"
            else encoder_learning_rate
        )
        for use_decay in (True, False):
            parameters = grouped[(family, use_decay)]
            if parameters:
                result.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay if use_decay else 0.0,
                        "group_name": f"{family}_{'decay' if use_decay else 'no_decay'}",
                    }
                )
    return result


def download_model(model_id: str, cache_dir: Path, revision: Optional[str]) -> Dict[str, str]:
    """Explicitly download a model snapshot and return pinned provenance."""
    try:
        from huggingface_hub import model_info, snapshot_download
    except ImportError as error:
        raise OptionalDependencyError(
            "huggingface_hub is required; activate the hm3d-semseg-train environment "
            "or install the train extra."
        ) from error
    info = model_info(model_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not resolve a commit for {model_id!r}")
    resolved_revision = str(info.sha)
    local = snapshot_download(
        repo_id=model_id,
        revision=resolved_revision,
        cache_dir=str(cache_dir),
    )
    card_data = getattr(info, "card_data", None)
    raw_license = getattr(card_data, "license", None) if card_data is not None else None
    license_name = str(raw_license) if raw_license is not None else "unknown"
    return {
        "model_id": model_id,
        "requested_revision": revision or "main",
        "resolved_revision": resolved_revision,
        "snapshot_path": str(Path(local).resolve()),
        "license": license_name,
    }
