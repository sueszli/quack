"""``assets/mjcf/patch_backlash.py`` generates the committed
``robot_*_backlash.xml`` models by patching the onshape-to-robot export.

The models are committed, so nothing normally re-runs the script — a regression
would only surface on the next Onshape re-export, months later. These tests
pin the generator against those committed artifacts.

The input is the NON-backlash sibling export (``robot_walk.xml`` for
``robot_walk_backlash.xml``). That is a genuinely independent input: each
``config_mjcf_<m>_backlash.json`` is identical to ``config_mjcf_<m>.json``
apart from appending the patch_backlash.py post_import_command, so the sibling
is exactly what the backlash pipeline feeds to the script. Reconstructing an
input by stripping backlash artifacts out of the committed output instead would
be self-confirming.
"""

import math
import re
import shutil
import subprocess
import sys

import pytest

from src.robot import MJCF_DIR

SCRIPT = MJCF_DIR / "patch_backlash.py"
# (backlash model, non-backlash sibling used as the pipeline input)
MODELS = [("robot_walk_backlash.xml", "robot_walk.xml"), ("robot_groundcontact_backlash.xml", "robot_groundcontact.xml"), ("robot_groundcontact_rollers_backlash.xml", "robot_groundcontact_rollers.xml")]
SERVO_COUNT = 14


def run_script(target, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(target), *args], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("committed,sibling", MODELS)
def test_regenerates_committed_model_byte_identically(committed, sibling, tmp_path):
    """Patching the non-backlash sibling must reproduce the committed model exactly."""
    target = tmp_path / committed
    shutil.copy(MJCF_DIR / sibling, target)

    proc = run_script(target, "--backlash-deg", "2.0")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert target.read_bytes() == (MJCF_DIR / committed).read_bytes()


@pytest.mark.parametrize("committed,sibling", MODELS)
def test_one_backlash_hinge_per_servo(committed, sibling):
    """Every servo gets exactly one hinge, on the same body/axis.

    A silently-skipped servo yields a model with fewer hinges than servos,
    which is a physics change no other test would catch.
    """
    text = (MJCF_DIR / committed).read_text()
    servos = re.findall(r'<joint\b[^>]*name="([^"]+)"[^>]*class="chosen_actuator"', text)
    hinges = re.findall(r'<joint\b[^>]*name="passive_([^"]+)_backlash"', text)

    assert len(servos) == SERVO_COUNT, f"{committed}: expected {SERVO_COUNT} servos, got {len(servos)}"
    assert hinges == servos, f"{committed}: hinge/servo mismatch"

    # Each hinge must share its servo's axis, else the play is on the wrong DOF.
    for name in servos:
        servo_axis = re.search(rf'<joint axis="([^"]*)"[^>]*name="{re.escape(name)}"', text)
        hinge_axis = re.search(rf'<joint axis="([^"]*)"[^>]*name="passive_{re.escape(name)}_backlash"', text)
        assert servo_axis and hinge_axis, f"{committed}: could not read axes for {name}"
        assert servo_axis.group(1) == hinge_axis.group(1), f"{committed}: axis mismatch on {name}"


def test_multiline_joint_aborts_instead_of_skipping(tmp_path):
    """A joint split across lines must fail loudly, not emit a short model.

    The scanner is line-based; before this guard such an export produced 13
    hinges for 14 servos and still exited 0.
    """
    target = tmp_path / "robot_walk_backlash.xml"
    text = (MJCF_DIR / "robot_walk.xml").read_text()
    m = re.search(r'^(\s*)<joint ([^>]*class="chosen_actuator"[^>]*)/>$', text, re.MULTILINE)
    assert m, "no single-line servo joint found to split"
    attrs = m.group(2).split(" ")
    split = f"{m.group(1)}<joint {' '.join(attrs[:2])}\n{m.group(1)}       {' '.join(attrs[2:])}/>"
    target.write_text(text.replace(m.group(0), split, 1))

    proc = run_script(target, "--backlash-deg", "2.0")

    assert proc.returncode == 1
    assert "spanning multiple lines" in proc.stdout


def test_refuses_to_run_twice(tmp_path):
    """The pipeline re-exports from Onshape each time; a second pass means
    something is wired wrong, and would double the modeled play."""
    target = tmp_path / "robot_walk_backlash.xml"
    shutil.copy(MJCF_DIR / "robot_walk.xml", target)

    assert run_script(target).returncode == 0
    after_first = target.read_bytes()
    proc = run_script(target)

    assert proc.returncode == 1
    assert "already contains backlash joints" in proc.stdout
    assert target.read_bytes() == after_first


def test_non_mjcf_input_is_not_written(tmp_path):
    target = tmp_path / "notmjcf.xml"
    target.write_text("<mujoco/>\n")

    proc = run_script(target)

    assert proc.returncode == 1
    assert "no <worldbody> found" in proc.stdout
    assert target.read_text() == "<mujoco/>\n"


def test_backlash_deg_scales_the_hinge_range(tmp_path):
    """--backlash-deg is TOTAL peak-to-peak play; the range is symmetric +/-deg/2."""
    target = tmp_path / "robot_walk_backlash.xml"
    shutil.copy(MJCF_DIR / "robot_walk.xml", target)

    assert run_script(target, "--backlash-deg", "5").returncode == 0

    match = re.search(r'range="(\S+) (\S+)" solreflimit', target.read_text())
    assert match, "no backlash hinge range found"
    lo, hi = match.groups()
    assert float(hi) == pytest.approx(math.radians(5) / 2)
    assert float(lo) == pytest.approx(-math.radians(5) / 2)
