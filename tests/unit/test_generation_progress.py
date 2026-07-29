from io import StringIO

import pytest

from hm3d_semseg.data.progress import DatasetGenerationProgress

pytestmark = pytest.mark.unit


def test_generation_progress_reports_scenes_samples_and_resume_counts() -> None:
    stream = StringIO()
    with DatasetGenerationProgress(file=stream) as progress:
        progress.start(total_scenes=2, total_samples=8, existing_samples=1)
        progress.scene_start(1, "scene-a")
        progress.samples_start("scene-a", 3)
        progress.sample_complete("existing")
        progress.sample_complete("accepted")
        progress.sample_complete("rejected")
        assert progress.accepted == 1
        assert progress.rejected == 1
        assert progress.existing == 1
        assert progress.stored_samples == 2
        progress.scene_complete()
        progress.scene_start(2, "scene-b")
        progress.samples_start("scene-b", 2)
        progress.sample_complete("accepted")
        progress.sample_complete("accepted")
        progress.scene_complete()

    output = stream.getvalue()
    assert "Scenes" in output
    assert "2/2" in output
    assert "stored=4/8" in output


def test_generation_progress_can_be_disabled() -> None:
    stream = StringIO()
    with DatasetGenerationProgress(enabled=False, file=stream) as progress:
        progress.start(total_scenes=1, total_samples=1, existing_samples=0)
        progress.scene_start(1, "scene")
        progress.samples_start("scene", 1)
        progress.sample_complete("accepted")
        progress.scene_complete()

    assert stream.getvalue() == ""


def test_generation_progress_rejects_unknown_sample_status() -> None:
    progress = DatasetGenerationProgress(enabled=False)
    with pytest.raises(ValueError, match="Unknown generation progress status"):
        progress.sample_complete("invalid")
