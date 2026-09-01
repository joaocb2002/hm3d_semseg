from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from hm3d_semseg.models.segformer import (
    lovasz_softmax_loss,
    parameter_groups,
    predict,
    segmentation_loss,
    segmentation_objective,
)

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


def test_zero_lovasz_weight_preserves_original_cross_entropy_exactly() -> None:
    logits = torch.randn(2, 41, 2, 3, requires_grad=True)
    targets = torch.tensor(
        [[[0, 1, 2], [3, 255, 40]], [[1, 1, 0], [2, 3, 4]]]
    )

    expected = segmentation_loss(logits, targets)
    losses = segmentation_objective(logits, targets)

    assert torch.equal(losses.objective, expected)
    assert torch.equal(losses.cross_entropy, expected)
    assert losses.lovasz.item() == 0.0


def test_lovasz_softmax_rewards_correct_known_class_rankings() -> None:
    targets = torch.tensor([[[0, 1], [2, 255]]])
    correct = torch.full((1, 3, 2, 2), -8.0, requires_grad=True)
    wrong = torch.full((1, 3, 2, 2), -8.0)
    with torch.no_grad():
        correct[0, 0, 0, 0] = 8.0
        correct[0, 1, 0, 1] = 8.0
        correct[0, 2, 1, 0] = 8.0
        wrong[0, 0, 0, 0] = 8.0
        wrong[0, 2, 0, 1] = 8.0
        wrong[0, 1, 1, 0] = 8.0

    correct_loss = lovasz_softmax_loss(correct, targets)
    wrong_loss = lovasz_softmax_loss(wrong, targets)

    assert correct_loss.item() < 1e-5
    assert wrong_loss.item() > 0.9
    correct_loss.backward()
    assert torch.isfinite(correct.grad).all()


def test_lovasz_native_resolution_keeps_unknown_as_a_negative() -> None:
    targets = torch.tensor(
        [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 255, 255], [2, 2, 255, 255]]]
    )
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)

    loss = lovasz_softmax_loss(
        logits,
        targets,
        include_unknown=False,
        resolution="native",
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
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
