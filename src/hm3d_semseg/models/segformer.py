"""Hugging Face SegFormer-B2 adapted to 41 project classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    classifier_learning_rate: float,
    weight_decay: float,
) -> list:
    classifier_parameters = []
    pretrained_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "decode_head.classifier" in name:
            classifier_parameters.append(parameter)
        else:
            pretrained_parameters.append(parameter)
    return [
        {
            "params": pretrained_parameters,
            "lr": encoder_learning_rate,
            "weight_decay": weight_decay,
            "group_name": "pretrained",
        },
        {
            "params": classifier_parameters,
            "lr": classifier_learning_rate,
            "weight_decay": weight_decay,
            "group_name": "classifier",
        },
    ]


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
    resolved_revision = info.sha
    local = snapshot_download(
        repo_id=model_id,
        revision=resolved_revision,
        cache_dir=str(cache_dir),
    )
    card_data = getattr(info, "card_data", None)
    license_name = getattr(card_data, "license", None) if card_data is not None else None
    return {
        "model_id": model_id,
        "requested_revision": revision or "main",
        "resolved_revision": resolved_revision,
        "snapshot_path": str(Path(local).resolve()),
        "license": license_name,
    }
