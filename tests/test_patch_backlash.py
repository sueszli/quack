import math
import re
import shutil
import subprocess
import sys

import pytest

from src.robot import MJCF_DIR

SCRIPT = MJCF_DIR / "patch_backlash.py"
MODELS = [("robot_walk_backlash.xml", "robot_walk.xml"), ("robot_groundcontact_backlash.xml", "robot_groundcontact.xml"), ("robot_groundcontact_rollers_backlash.xml", "robot_groundcontact_rollers.xml")]
SERVO_COUNT = 14


def run_script(target, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(target), *args], capture_output=True, text=True, check=False)


@pytest.mark.parametrize("committed,sibling", MODELS)
def test_patching_the_non_backlash_sibling_regenerates_the_committed_model_byte_identically(committed, sibling, tmp_path):
    target = tmp_path / committed
    shutil.copy(MJCF_DIR / sibling, target)

    proc = run_script(target, "--backlash-deg", "2.0")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert target.read_bytes() == (MJCF_DIR / committed).read_bytes()


@pytest.mark.parametrize("committed,sibling", MODELS)
def test_every_servo_has_exactly_one_backlash_hinge_on_the_same_axis(committed, sibling):
    text = (MJCF_DIR / committed).read_text()
    servos = re.findall(r'<joint\b[^>]*name="([^"]+)"[^>]*class="chosen_actuator"', text)
    hinges = re.findall(r'<joint\b[^>]*name="passive_([^"]+)_backlash"', text)

    assert len(servos) == SERVO_COUNT, f"{committed}: expected {SERVO_COUNT} servos, got {len(servos)}"
    assert hinges == servos, f"{committed}: hinge/servo mismatch"

    for name in servos:
        servo_axis = re.search(rf'<joint axis="([^"]*)"[^>]*name="{re.escape(name)}"', text)
        hinge_axis = re.search(rf'<joint axis="([^"]*)"[^>]*name="passive_{re.escape(name)}_backlash"', text)
        assert servo_axis and hinge_axis, f"{committed}: could not read axes for {name}"
        assert servo_axis.group(1) == hinge_axis.group(1), f"{committed}: play is on the wrong DOF for {name}"


def test_a_joint_spanning_multiple_lines_aborts_instead_of_emitting_a_short_model(tmp_path):
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


def test_running_twice_refuses_rather_than_doubling_the_modeled_play(tmp_path):
    target = tmp_path / "robot_walk_backlash.xml"
    shutil.copy(MJCF_DIR / "robot_walk.xml", target)

    assert run_script(target).returncode == 0
    after_first = target.read_bytes()
    proc = run_script(target)

    assert proc.returncode == 1
    assert "already contains backlash joints" in proc.stdout
    assert target.read_bytes() == after_first


def test_non_mjcf_input_is_rejected_without_being_written(tmp_path):
    target = tmp_path / "notmjcf.xml"
    target.write_text("<mujoco/>\n")

    proc = run_script(target)

    assert proc.returncode == 1
    assert "no <worldbody> found" in proc.stdout
    assert target.read_text() == "<mujoco/>\n"


def test_backlash_deg_is_total_play_so_the_hinge_range_is_half_of_it_each_way(tmp_path):
    target = tmp_path / "robot_walk_backlash.xml"
    shutil.copy(MJCF_DIR / "robot_walk.xml", target)

    assert run_script(target, "--backlash-deg", "5").returncode == 0

    match = re.search(r'range="(\S+) (\S+)" solreflimit', target.read_text())
    assert match, "no backlash hinge range found"
    lo, hi = match.groups()
    assert float(hi) == pytest.approx(math.radians(5) / 2)
    assert float(lo) == pytest.approx(-math.radians(5) / 2)
