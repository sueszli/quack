import re

import pytest
from mjlab.utils.os import get_checkpoint_path

from src.export import CHECKPOINT_PATTERN


def _run_dir(root, name):
    d = root / "logs" / "rsl_rl" / "velocity" / name
    d.mkdir(parents=True)
    return d


def test_latest_fallback_skips_tensorboard_artifacts(tmp_path):
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    (run / "model_250.pt").touch()
    (run / "model_3000.pt").touch()
    (run / "events.out.tfevents.1788978725.host.1.0").touch()
    (run / "params").mkdir()
    (run / "git").mkdir()

    resolved = get_checkpoint_path(run.parent.parent / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.name == "model_3000.pt"


def test_latest_fallback_orders_numerically_not_lexically(tmp_path):
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    for n in (0, 250, 900, 1000, 10000):
        (run / f"model_{n}.pt").touch()
    (run / "events.out.tfevents.1788978725.host.1.0").touch()

    resolved = get_checkpoint_path(run.parent.parent / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.name == "model_10000.pt"


def test_latest_fallback_picks_newest_run_dir(tmp_path):
    old = _run_dir(tmp_path, "2025-01-01_00-00-00_velocity")
    (old / "model_9000.pt").touch()
    new = _run_dir(tmp_path, "2026-06-01_00-00-00_velocity")
    (new / "model_100.pt").touch()

    resolved = get_checkpoint_path(tmp_path / "logs" / "rsl_rl" / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.parent.name == "2026-06-01_00-00-00_velocity"
    assert resolved.name == "model_100.pt"


def test_pattern_rejects_non_checkpoints():
    assert re.match(CHECKPOINT_PATTERN, "model_3000.pt")
    assert not re.match(CHECKPOINT_PATTERN, "events.out.tfevents.1788978725.host.1.0")
    assert not re.match(CHECKPOINT_PATTERN, "params")
    assert not re.match(CHECKPOINT_PATTERN, "model_best.pt")
    assert not re.match(CHECKPOINT_PATTERN, "xmodel_3000.pt")
    assert not re.match(CHECKPOINT_PATTERN, "model_3000.pt.partial")


def test_explicit_checkpoint_300_does_not_match_model_3000(tmp_path):
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    (run / "model_3000.pt").touch()

    with pytest.raises(ValueError, match="No checkpoint found"):
        get_checkpoint_path(tmp_path / "logs" / "rsl_rl" / "velocity", checkpoint=re.escape("model_300.pt"))
