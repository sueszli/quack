from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MDP = REPO / "src" / "task_mdp.py"
SKIP_DIRS = {".git", "__pycache__", ".venv", "logs", "node_modules"}


def _repo_text_without_mdp() -> str:
    chunks = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path == MDP:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def test_no_unreferenced_public_functions() -> None:
    source = MDP.read_text(encoding="utf-8")
    outside = _repo_text_without_mdp()

    dead = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        word = re.compile(r"\b" + re.escape(node.name) + r"\b")
        if word.search(outside):
            continue
        if len(word.findall(source)) - 1 > 0:
            continue
        dead.append(f"{node.name} (line {node.lineno})")

    assert not dead, "task_mdp.py functions referenced nowhere in the repo — delete them or wire them into a task:\n  " + "\n  ".join(sorted(dead))
