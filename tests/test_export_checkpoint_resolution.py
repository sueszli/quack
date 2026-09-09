"""Checkpoint resolution in src/export.py.

Since wandb was dropped, `export`/`publish` resolve checkpoints purely from the local
log tree, so the resolution rules are now load-bearing and deserve a test.

The trap these lock down: mjlab's `get_checkpoint_path` defaults to `checkpoint=".*"`,
which matches EVERY entry in a run directory, and its sort is alphabetical. With
`logger="tensorboard"` there is always an `events.out.tfevents.*` file (and a `params/`
directory) sitting next to the checkpoints, and both sort after `model_*.pt` — so the
default pattern silently hands back the event file and `torch.load` dies on it.
"""

import re

import pytest
from mjlab.utils.os import get_checkpoint_path

from src.export import CHECKPOINT_PATTERN


def _run_dir(root, name):
    d = root / "logs" / "rsl_rl" / "velocity" / name
    d.mkdir(parents=True)
    return d


def test_latest_fallback_skips_tensorboard_artifacts(tmp_path):
    """The no-flag fallback must return a checkpoint, not the tfevents file or params/."""
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    (run / "model_250.pt").touch()
    (run / "model_3000.pt").touch()
    (run / "events.out.tfevents.1788978725.host.1.0").touch()
    (run / "params").mkdir()
    (run / "git").mkdir()

    resolved = get_checkpoint_path(run.parent.parent / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.name == "model_3000.pt"


def test_latest_fallback_orders_numerically_not_lexically(tmp_path):
    """model_900 < model_1000 < model_10000 — mjlab's zero-pad sort must hold."""
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    for n in (0, 250, 900, 1000, 10000):
        (run / f"model_{n}.pt").touch()
    (run / "events.out.tfevents.1788978725.host.1.0").touch()

    resolved = get_checkpoint_path(run.parent.parent / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.name == "model_10000.pt"


def test_latest_fallback_picks_newest_run_dir(tmp_path):
    """Run dirs are datetime-prefixed, so alphabetical == chronological."""
    old = _run_dir(tmp_path, "2025-01-01_00-00-00_velocity")
    (old / "model_9000.pt").touch()
    new = _run_dir(tmp_path, "2026-06-01_00-00-00_velocity")
    (new / "model_100.pt").touch()

    resolved = get_checkpoint_path(tmp_path / "logs" / "rsl_rl" / "velocity", checkpoint=CHECKPOINT_PATTERN)
    assert resolved.parent.name == "2026-06-01_00-00-00_velocity"
    assert resolved.name == "model_100.pt"


def test_pattern_rejects_non_checkpoints():
    assert re.search(CHECKPOINT_PATTERN, "model_3000.pt")
    assert not re.search(CHECKPOINT_PATTERN, "events.out.tfevents.1788978725.host.1.0")
    assert not re.search(CHECKPOINT_PATTERN, "params")
    assert not re.search(CHECKPOINT_PATTERN, "model_best.pt")


def test_explicit_checkpoint_is_exact(tmp_path):
    """`--checkpoint 300` must not match model_3000.pt."""
    run = _run_dir(tmp_path, "2026-01-01_00-00-00_velocity")
    (run / "model_3000.pt").touch()

    with pytest.raises(ValueError, match="No checkpoint found"):
        get_checkpoint_path(tmp_path / "logs" / "rsl_rl" / "velocity", checkpoint=re.escape("model_300.pt"))
