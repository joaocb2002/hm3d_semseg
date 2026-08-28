from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from hm3d_semseg.models.segformer import parameter_groups, predict, segmentation_loss

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


def test_optimizer_groups_exclude_bias_and_norm_from_weight_decay() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1),
                torch.nn.BatchNorm2d(4),
            )
            self.decode_head = torch.nn.Module()
            self.decode_head.projection = torch.nn.Conv2d(4, 4, 1)
            self.decode_head.classifier = torch.nn.Conv2d(4, 41, 1)

    groups = parameter_groups(
        Model(),
        6e-5,
        6e-4,
        0.01,
        entire_decode_head=True,
        exclude_one_dimensional_from_decay=True,
    )
    by_name = {group["group_name"]: group for group in groups}

    assert set(by_name) == {
        "pretrained_decay",
        "pretrained_no_decay",
        "decode_head_decay",
        "decode_head_no_decay",
    }
    assert by_name["pretrained_decay"]["weight_decay"] == 0.01
    assert by_name["decode_head_decay"]["weight_decay"] == 0.01
    assert by_name["pretrained_no_decay"]["weight_decay"] == 0.0
    assert by_name["decode_head_no_decay"]["weight_decay"] == 0.0
    assert by_name["pretrained_decay"]["lr"] == 6e-5
    assert by_name["decode_head_decay"]["lr"] == 6e-4

    legacy = parameter_groups(Model(), 6e-5, 6e-4, 0.01)
    legacy_by_name = {group["group_name"]: group for group in legacy}
    assert set(legacy_by_name) == {"pretrained_decay", "decode_head_decay"}
    projection = legacy_by_name["pretrained_decay"]["params"]
    assert any(tuple(parameter.shape) == (4, 4, 1, 1) for parameter in projection)
