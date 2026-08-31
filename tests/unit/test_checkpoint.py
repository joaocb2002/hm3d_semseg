import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hm3d_semseg.training.checkpoint import (
    atomic_torch_save,
    load_training_state,
    save_checkpoint,
    update_checkpoint_progress,
)

pytestmark = pytest.mark.unit


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "state.pt"
    atomic_torch_save(model.state_dict(), path)
    restored = torch.nn.Linear(2, 3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_persists_best_development_loss(tmp_path: Path) -> None:
    class FakeModel:
        def save_pretrained(self, target: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization
            (target / "model.safetensors").write_bytes(b"weights")

    camera_profile = tmp_path / "camera.yaml"
    camera_profile.write_text("schema_version: '1.0'\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        FakeModel(),
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=3,
        step=40,
        primary_metric=0.4,
        camera_profile_path=camera_profile,
        best_development_loss=0.8,
    )

    assert load_training_state(checkpoint)["best_development_loss"] == pytest.approx(0.8)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    assert metadata["best_development_loss"] == pytest.approx(0.8)

    update_checkpoint_progress(
        checkpoint,
        primary_metric=0.45,
        epochs_without_improvement=2,
        best_development_loss=0.7,
    )
    state = load_training_state(checkpoint)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    assert state["best_development_loss"] == pytest.approx(0.7)
    assert metadata["best_development_loss"] == pytest.approx(0.7)
