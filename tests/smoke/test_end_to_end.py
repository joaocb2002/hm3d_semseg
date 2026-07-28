from __future__ import annotations

import os
from pathlib import Path

import pytest

from hm3d_semseg.config import load_config
from hm3d_semseg.diagnostics.smoke import run_smoke_test

pytestmark = [pytest.mark.smoke, pytest.mark.slow, pytest.mark.gpu]


def test_full_smoke_workflow() -> None:
    if os.environ.get("RUN_HM3D_SMOKE") != "1":
        pytest.skip("Set RUN_HM3D_SMOKE=1 to authorize the costly smoke workflow")
    local = os.environ.get("HM3D_SEMSEG_LOCAL_CONFIG")
    if not local or not Path(local).is_file():
        pytest.skip("Set HM3D_SEMSEG_LOCAL_CONFIG to a valid local config")
    report = run_smoke_test(load_config(local_config=Path(local)))
    assert report["validation"]["samples"] >= 2
    assert report["training"]["global_steps"] >= 2
    assert report["inference"]["probabilities_saved"] is False
