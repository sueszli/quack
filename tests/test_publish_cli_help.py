from __future__ import annotations

import ast
import dataclasses
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import tyro

from src.publish_cli import PublishConfig

_ROOT = Path(__file__).resolve().parents[1]

# PublishConfig documents its fields with `#` comments above each field rather
# than with attribute docstrings. Tyro reads those preceding comments to build
# `publish --help`, so the help text depends on comment PLACEMENT, not on a
# language feature: a reformat that moves, reflows or drops a comment would
# silently gut the CLI help without breaking anything else. These tests pin
# that down.


def _help_text() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit):
        tyro.cli(PublishConfig, args=["--help"])
    # Undo the box-drawing wrap so assertions can match on prose that tyro
    # split across lines.
    text = buf.getvalue()
    text = re.sub(r"[│╭╮╰╯─]", " ", text)
    return re.sub(r"\s+", " ", text)


def test_every_field_is_documented_in_help():
    # The failure mode this guards: a comment drifts away from its field, so
    # tyro emits the flag with no description at all.
    help_text = _help_text()
    for field in dataclasses.fields(PublishConfig):
        flag = "--" + field.name.replace("_", "-")
        assert flag in help_text or f"--{field.name}" in help_text, f"{field.name} missing from --help"


@pytest.mark.parametrize(("flag", "snippet"), [("--repo", "Hub repo id"), ("--kind", "runs `duration_s` and comes back on its own"), ("--task", "Task id to export from"), ("--onnx", "Validated, not re-exported"), ("--dry-run", "and stop"), ("--device", "Default: cuda:0 if available")])
def test_field_help_text_survives(flag, snippet):
    assert snippet in _help_text(), f"{flag} lost its help text"


def test_section_headers_do_not_leak_into_field_help():
    # `# -- where it goes` and friends are section headers, not field docs. They
    # need a blank line after them, otherwise tyro glues the header onto the
    # next field's description ("-- where it goes Hub repo id, ...").
    help_text = _help_text()
    for header in ("-- where it goes", "-- where the weights come from", "-- what the manifest says", "-- how"):
        assert header not in help_text, f"section header {header!r} leaked into --help"


def _python_files():
    skip = {".git", ".venv", "__pycache__", "logs", "data", "weights"}
    return [p for p in sorted(_ROOT.rglob("*.py")) if not skip & set(p.parts)]


def test_the_project_contains_no_docstrings():
    # Project-wide convention: documentation is `#` comments, never docstrings.
    # This covers module, class and function docstrings AND bare string
    # statements such as tyro attribute docstrings, in every Python file.
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for stmt in body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    offenders.append(f"{path.relative_to(_ROOT)}:{stmt.lineno}")
    assert not offenders, "docstrings must be `#` comments:\n  " + "\n  ".join(offenders)


def test_the_audit_scans_the_whole_project():
    # Guards the guard: if the glob or skip-list ever stops finding files, the
    # audit above would pass vacuously.
    files = _python_files()
    assert len(files) > 40, f"only found {len(files)} python files"
    names = {p.name for p in files}
    assert {"publish_cli.py", "task_mdp.py", "robot.py"} <= names
