"""The tyro CLI configs must keep the flags and defaults the argparse ones had.

`infer` is the deployment rehearsal and `duck-body` serves a body to the real
`robotd`, so a silently renamed flag or drifted default is a sim2real footgun,
not a cosmetic bug. These tests pin the flag surface (including the `-s` alias
and the three-mode `--delay`) and the validation errors.
"""

import dataclasses

import pytest
import tyro

from src.infer import InferConfig
from src.sim_body_server import ServerConfig

# name -> default, transcribed from the argparse parsers these dataclasses replaced.
_INFER_FLAGS = {"roller": False, "scene": None, "walking": None, "standing": None, "ground_pick": None, "sit": None, "sitstand": None, "slope": None, "kick_left": None, "kick_right": None, "roulade": None, "kick_duration": 3.0, "roulade_duration": 2.0, "lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": 0.0, "action_scale": 1.0, "raw_accelerometer": False, "delay": None, "debug": False, "save_csv": None, "record": None, "switch_threshold": 0.05, "ground_pick_period": 4.0, "new_cmd_obs": False, "no_bam": False, "vin": 7.4, "vin_drop_gain": 0.1, "kp_fw": 200.0, "current_limit": 0.0, "foot_friction": None, "foot_solref": None}

_SERVER_FLAGS = {"scene", "ducks", "host", "port", "headless", "cameras", "frame_port", "camera_fps", "limp", "keyframe"}


def _defaults(cls):
    return {f.name: f.default for f in dataclasses.fields(cls)}


def test_infer_flag_surface_is_unchanged():
    assert set(_defaults(InferConfig)) == set(_INFER_FLAGS)


@pytest.mark.parametrize(("name", "expected"), sorted(_INFER_FLAGS.items(), key=lambda kv: kv[0]))
def test_infer_defaults_are_unchanged(name, expected):
    assert _defaults(InferConfig)[name] == expected


def test_server_flag_surface_is_unchanged():
    assert set(_defaults(ServerConfig)) == _SERVER_FLAGS


def test_standing_keeps_its_short_alias():
    """`-s` is the documented shorthand for --standing."""
    assert tyro.cli(InferConfig, args=["-s", "stand.onnx"]).standing == "stand.onnx"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),  # flag absent
        (["--delay"], []),  # bare: main() maps this to lag 1-2
        (["--delay", "3"], [3]),  # one value: min == max
        (["--delay", "1", "2"], [1, 2]),  # explicit min/max
    ],
)
def test_delay_keeps_its_three_arities(argv, expected):
    """argparse used nargs='*'; main() still branches on len(delay)."""
    assert tyro.cli(InferConfig, args=[*argv]).delay == expected


def test_bam_is_on_by_default():
    """Policies are trained against BAM; --no-bam must be opt-in."""
    assert tyro.cli(InferConfig, args=[]).no_bam is False


@pytest.mark.parametrize(("argv", "fragment"), [([], "At least one of --walking, --standing or --sitstand"), (["--sitstand", "s.onnx"], "add --new-cmd-obs"), (["--walking", "w.onnx", "--kick-left", "k.onnx"], "add --new-cmd-obs"), (["--walking", "w.onnx", "--roulade", "r.onnx", "--new-cmd-obs", "--roller"], "not the roller model")])
def test_invalid_combinations_are_rejected(argv, fragment, monkeypatch):
    from src import infer

    monkeypatch.setattr("sys.argv", ["infer", *argv])
    with pytest.raises(SystemExit) as excinfo:
        infer.main()
    assert fragment in str(excinfo.value)
