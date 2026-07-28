from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from hm3d_semseg.models.segformer import predict, segmentation_loss

pytestmark = pytest.mark.unit


class FakeSegformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Conv2d(3, 41, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        logits = self.classifier(pixel_values[:, :, ::2, ::2])
        return SimpleNamespace(logits=logits)


def test_model_output_is_41_way_distribution() -> None:
    model = FakeSegformer()
    pixels = torch.randn(2, 3, 8, 6)
    output = predict(model, pixels, output_size=(8, 6))
    assert output.logits.shape == (2, 41, 8, 6)
    assert output.probabilities.shape == (2, 41, 8, 6)
    assert output.labels.shape == (2, 8, 6)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2, 8, 6), atol=1e-6)


def test_cross_entropy_gets_raw_logits_and_ignores_255() -> None:
    logits = torch.zeros(1, 41, 2, 2, requires_grad=True)
    targets = torch.tensor([[[0, 1], [255, 40]]])
    loss = segmentation_loss(logits, targets)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(41.0)).item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
