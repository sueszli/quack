from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np

SEED = 41

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATA_DIR = PROJECT_ROOT / "data"


def weights_path(*parts: str | os.PathLike[str]) -> Path:
    p = WEIGHTS_DIR.joinpath(*[str(x) for x in parts])
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def data_path(*parts: str | os.PathLike[str]) -> Path:
    p = DATA_DIR.joinpath(*[str(x) for x in parts])
    (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def use_local_storage(verbose: bool = False) -> dict[str, str]:
    layout: dict[str, Path] = {"HF_HOME": WEIGHTS_DIR / "hf", "HUGGINGFACE_HUB_CACHE": WEIGHTS_DIR / "hf" / "hub", "TORCH_HOME": WEIGHTS_DIR / "torch", "WANDB_ARTIFACT_DIR": WEIGHTS_DIR / "wandb_artifacts", "WANDB_CACHE_DIR": WEIGHTS_DIR / "wandb_cache", "WANDB_DIR": DATA_DIR / "wandb", "MUJOCO_GL_CACHE": DATA_DIR / "cache" / "mujoco", "WARP_CACHE_PATH": DATA_DIR / "cache" / "warp", "TRITON_CACHE_DIR": DATA_DIR / "cache" / "triton", "XDG_CACHE_HOME": DATA_DIR / "cache" / "xdg", "MPLCONFIGDIR": DATA_DIR / "cache" / "matplotlib"}
    applied: dict[str, str] = {}
    for key, path in layout.items():
        if os.environ.get(key):
            continue
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        applied[key] = str(path)
    if verbose and applied:
        for key, value in sorted(applied.items()):
            print(f"[storage] {key}={value}")
    return applied


def seed_from_string(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def set_seed(seed: int = SEED, deterministic: bool = False) -> int:
    seed = int(seed) % (2**32)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    try:
        import warp as wp

        wp.rand_init(seed)
    except Exception:
        pass

    return seed


use_local_storage()
set_seed()
