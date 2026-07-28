from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hm3d_semseg.training.checkpoint import atomic_torch_save

pytestmark = pytest.mark.unit


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "state.pt"
    atomic_torch_save(model.state_dict(), path)
    restored = torch.nn.Linear(2, 3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
