import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.utils import DATA_DIR, PROJECT_ROOT, SEED, WEIGHTS_DIR, data_path, seed_from_string, set_seed, use_local_storage


def _expected_first_draw():
    random.seed(SEED)
    return random.random()


def _draw():
    return (random.random(), float(np.random.rand()))


def test_set_seed_makes_python_and_numpy_reproducible():
    set_seed(1234)
    first = _draw()
    set_seed(1234)
    assert _draw() == first

    set_seed(4321)
    assert _draw() != first


def test_set_seed_seeds_torch():
    torch = pytest.importorskip("torch")
    set_seed(7)
    a = torch.randn(8)
    set_seed(7)
    assert torch.equal(a, torch.randn(8))


def test_set_seed_returns_seed_and_sets_pythonhashseed():
    assert set_seed(99) == 99
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_pythonhashseed_is_exported_before_interpreter_start():
    env = dict(os.environ, PYTHONHASHSEED="0")
    code = "print(hash('microduck'))"
    out = [subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout for _ in range(2)]
    assert out[0] == out[1]


def test_deterministic_mode_sets_cublas_workspace():
    pytest.importorskip("torch")
    set_seed(0, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    set_seed(0)


def test_import_seeds_and_routes_storage_without_any_call():
    code = "import src.utils, os, random; print(random.random()); print(os.environ['HF_HOME'])"
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT).stdout for _ in range(2)]
    assert outs[0] == outs[1]
    draw, hf_home = outs[0].splitlines()[-2:]
    assert float(draw) == pytest.approx(_expected_first_draw())
    assert Path(hf_home) == WEIGHTS_DIR / "hf"


def test_default_seed_is_the_project_seed():
    assert set_seed() == SEED == 41


def test_seed_from_string_is_stable_and_in_range():
    assert seed_from_string("velocity-flat") == seed_from_string("velocity-flat")
    assert seed_from_string("a") != seed_from_string("b")
    assert 0 <= seed_from_string("x") < 2**32


def test_storage_dirs_live_inside_the_repo():
    assert WEIGHTS_DIR == PROJECT_ROOT / "weights"
    assert DATA_DIR == PROJECT_ROOT / "data"
    p = data_path("pytest_tmp", "probe.txt")
    assert p.parent.is_dir() and PROJECT_ROOT in p.parents


def test_use_local_storage_points_caches_into_the_repo():
    for key in ("HF_HOME", "TORCH_HOME", "WANDB_DIR", "WARP_CACHE_PATH"):
        os.environ.pop(key, None)
    use_local_storage()
    for key in ("HF_HOME", "TORCH_HOME", "WANDB_DIR", "WARP_CACHE_PATH"):
        assert PROJECT_ROOT in Path(os.environ[key]).parents, key


def test_use_local_storage_respects_explicit_overrides():
    os.environ["TORCH_HOME"] = "/tmp/explicit-torch-home"
    try:
        use_local_storage()
        assert os.environ["TORCH_HOME"] == "/tmp/explicit-torch-home"
    finally:
        del os.environ["TORCH_HOME"]


def test_storage_is_gitignored():
    ignored = (PROJECT_ROOT / ".gitignore").read_text()
    assert "weights/" in ignored
    assert "data/" in ignored
