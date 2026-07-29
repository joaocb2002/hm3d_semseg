from io import StringIO

import pytest

from hm3d_semseg.training.progress import TrainingProgress

pytestmark = pytest.mark.unit


def test_training_progress_reports_setup_steps_and_live_metrics() -> None:
    stream = StringIO()
    progress = TrainingProgress(file=stream)
    progress.message("Validating training dataset")
    progress.start(
        run="/runs/tiny",
        device="cuda:0",
        samples=4,
        epochs=2,
        batch_size=2,
        batches_per_epoch=2,
        steps_per_epoch=2,
        total_steps=4,
        completed_steps=0,
        gradient_accumulation_steps=1,
        amp=False,
        trainable_parameters=75,
        total_parameters=100,
        encoder_learning_rate=1e-4,
        classifier_learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=1,
    )
    progress.step(epoch=0, loss=1.25, learning_rate=1e-4, samples_per_second=3.5)
    progress.phase(epoch=0, name="checkpoint")
    progress.close()

    output = stream.getvalue()
    assert "Validating training dataset" in output
    assert "samples=4" in output
    assert "total_steps=4" in output
    assert "75 trainable / 100 total (75.0% trainable)" in output
    assert "encoder_lr=1.00e-04" in output
    assert "classifier_lr=1.00e-03" in output
    assert "1/4" in output
    assert "loss=1.2500" in output
    assert "phase=checkpoint" in output


def test_training_progress_can_be_disabled() -> None:
    stream = StringIO()
    progress = TrainingProgress(enabled=False, file=stream)
    progress.message("Validating training dataset")
    progress.start(
        run="/runs/tiny",
        device="cpu",
        samples=4,
        epochs=1,
        batch_size=2,
        batches_per_epoch=2,
        steps_per_epoch=2,
        total_steps=2,
        completed_steps=0,
        gradient_accumulation_steps=1,
        amp=False,
        trainable_parameters=100,
        total_parameters=100,
        encoder_learning_rate=1e-4,
        classifier_learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
    )
    progress.step(epoch=0, loss=1.0, learning_rate=1e-4, samples_per_second=1.0)
    progress.phase(epoch=0, name="checkpoint")
    progress.close()

    assert stream.getvalue() == ""
