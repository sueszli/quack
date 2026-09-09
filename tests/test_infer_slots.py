from __future__ import annotations

import json

import pytest

from src import infer as ip
from src import publish_manifest as pm


def test_every_slot_has_a_flag_and_a_unique_dest():
    dests = [s.dest for s in ip.POLICY_SLOTS]
    assert len(dests) == len(set(dests))
    for slot in ip.POLICY_SLOTS:
        assert slot.flag.startswith("--")
        assert slot.dest == slot.flag[2:].replace("-", "_")


def test_hotkeys_are_lowercase_single_keys_and_unambiguous():
    empty = _FakePolicy()
    for slot in ip.POLICY_SLOTS:
        if slot.hotkey is None:
            continue
        assert len(slot.hotkey) == 1 and slot.hotkey.islower()
        assert slot.action, f"{slot.name} binds {slot.hotkey!r} to nothing"
    for key in {s.hotkey for s in ip.POLICY_SLOTS if s.hotkey}:
        claimants = ip.slots_with_hotkey(key)
        if len(claimants) > 1:
            assert all(ip.slot_is_loaded(empty, s) is False for s in claimants), f"{key!r} is shared but a claimant's loaded-state is unreadable"


def test_policy_hotkeys_do_not_collide_with_the_fixed_keys():
    reserved = {"t", "q", "h", "b", "p", " ", "up", "down", "left", "right"}
    assert not ({s.hotkey for s in ip.POLICY_SLOTS if s.hotkey} & reserved)


def test_episodic_slots_are_timed():
    for slot in ip.POLICY_SLOTS:
        if slot.kind == "episodic":
            assert slot.duration_dest, f"{slot.name} is episodic but nothing times it"
            assert slot.duration_dest in ip.DURATION_FLAGS_BY_DEST
        else:
            assert slot.duration_dest is None


def test_slot_vocabulary_is_the_manifest_vocabulary():
    for slot in ip.POLICY_SLOTS:
        assert slot.kind in pm.KINDS
        assert slot.encoding in pm.ENCODINGS
        if slot.daemon_slot is not None:
            assert slot.daemon_slot in pm.SLOTS


def test_behavior_slots_are_the_timed_session_swaps():
    assert {s.name for s in ip.BEHAVIOR_SLOTS} == {"kick_left", "kick_right", "roulade"}
    for slot in ip.BEHAVIOR_SLOTS:
        assert slot.needs_cmd_obs and slot.kind == "episodic" and slot.encoding == "constant"
        assert slot.walk_model_only, f"{slot.name} is trained on the walking robot, not the rollers"


def test_ball_scene_is_requested_by_exactly_the_kick_slots():
    assert {s.name for s in ip.POLICY_SLOTS if s.needs_ball} == {"kick_left", "kick_right"}


def test_y_group_prefers_sit_over_slope():
    claimants = ip.slots_with_hotkey("y")
    assert [s.name for s in claimants] == ["sit", "sitstand", "slope"]
    assert claimants[-1].action == "toggle_slope_mode"


class _FakePolicy:
    def __init__(self, behaviors=(), **sessions):
        for slot in ip.POLICY_SLOTS:
            if slot.session_attr:
                setattr(self, slot.session_attr, None)
        self.behavior_sessions = {name: object() for name in behaviors}
        for name, value in sessions.items():
            setattr(self, name, value)
        self.calls: list[tuple[str, str | None]] = []

    def _record(self, name):
        def call(arg=None):
            self.calls.append((name, arg))

        return call

    def __getattr__(self, name):
        if name in {s.action for s in ip.POLICY_SLOTS if s.action}:
            return self._record(name)
        raise AttributeError(name)


def test_dispatch_routes_each_hotkey_to_its_action():
    policy = _FakePolicy(behaviors=("kick_left", "kick_right", "roulade"), walking_session=object(), ground_pick_session=object())
    for key, expected in (("g", ("trigger_ground_pick", None)), ("k", ("trigger_behavior", "kick_left")), ("l", ("trigger_behavior", "kick_right")), ("r", ("trigger_behavior", "roulade"))):
        policy.calls.clear()
        assert ip.dispatch_hotkey(policy, key) is True
        assert policy.calls == [expected]


def test_dispatch_y_picks_the_loaded_slot():
    with_sit = _FakePolicy(sit_session=object(), slope_session=object())
    assert ip.dispatch_hotkey(with_sit, "y") is True
    assert with_sit.calls == [("toggle_sit", None)]

    with_slope = _FakePolicy(slope_session=object())
    assert ip.dispatch_hotkey(with_slope, "y") is True
    assert with_slope.calls == [("toggle_slope_mode", None)]

    empty = _FakePolicy()
    assert ip.dispatch_hotkey(empty, "y") is True
    assert empty.calls == [("toggle_slope_mode", None)]


def test_dispatch_declines_keys_it_does_not_own():
    policy = _FakePolicy()
    for key in ("t", "q", "h", "b", "p", " ", "up"):
        assert ip.dispatch_hotkey(policy, key) is False
    assert policy.calls == []


def _repo(tmp_path, manifest: dict, onnx_name: str = pm.POLICY_FILE):
    directory = tmp_path / "repo"
    directory.mkdir(exist_ok=True)
    (directory / onnx_name).write_bytes(b"not a real graph")  # never opened by the resolver
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


ROULADE_MANIFEST = pm.build_manifest(name="roulade", kind="episodic", description="forward roll", duration_s=2.5, slot="roulade")


def test_plain_onnx_path_is_untouched(tmp_path):
    slot = ip.POLICY_SLOTS_BY_NAME["roulade"]
    resolved = ip.resolve_policy_argument(slot, "out.onnx", default_duration=2.0, new_cmd_obs=True)
    assert resolved.onnx_path == "out.onnx"
    assert resolved.duration_s == 2.0
    assert resolved.manifest is None and resolved.source == "flag"


def test_directory_without_a_manifest_is_not_a_repo(tmp_path):
    (tmp_path / "plain").mkdir()
    assert ip.find_policy_manifest(str(tmp_path / "plain")) is None
    assert ip.find_policy_manifest("out.onnx") is None


def test_manifest_supplies_the_network_and_the_duration(tmp_path):
    directory = _repo(tmp_path, ROULADE_MANIFEST)
    slot = ip.POLICY_SLOTS_BY_NAME["roulade"]
    resolved = ip.resolve_policy_argument(slot, str(directory), default_duration=2.0, new_cmd_obs=True)
    assert resolved.onnx_path == str(directory / pm.POLICY_FILE)
    assert resolved.duration_s == 2.5  # the manifest's, not --roulade-duration's
    assert resolved.manifest["name"] == "roulade"
    assert resolved.source.endswith("manifest.json")


def test_manifest_repo_with_one_oddly_named_onnx_still_resolves(tmp_path):
    directory = _repo(tmp_path, ROULADE_MANIFEST, onnx_name="model.onnx")
    slot = ip.POLICY_SLOTS_BY_NAME["roulade"]
    resolved = ip.resolve_policy_argument(slot, str(directory), default_duration=2.0, new_cmd_obs=True)
    assert resolved.onnx_path == str(directory / "model.onnx")


def test_perpetual_manifest_keeps_the_flag_duration(tmp_path):
    directory = _repo(tmp_path, pm.build_manifest(name="walk", kind="perpetual", description="a gait", slot="walk"))
    resolved = ip.resolve_policy_argument(ip.POLICY_SLOTS_BY_NAME["walking"], str(directory), default_duration=None, new_cmd_obs=True)
    assert resolved.duration_s is None


def test_61d_manifest_without_new_cmd_obs_is_refused(tmp_path):
    directory = _repo(tmp_path, ROULADE_MANIFEST)
    with pytest.raises(AssertionError, match="new-cmd-obs"):
        ip.resolve_policy_argument(ip.POLICY_SLOTS_BY_NAME["roulade"], str(directory), default_duration=2.0, new_cmd_obs=False)


def test_a_policy_set_is_refused(tmp_path):
    directory = _repo(tmp_path, {"schema_version": 2, "policies": [{"file": "a.onnx", "name": "a", "kind": "episodic", "duration_s": 1.0}]})
    with pytest.raises(AssertionError, match="policy SET"):
        ip.resolve_policy_argument(ip.POLICY_SLOTS_BY_NAME["roulade"], str(directory), default_duration=2.0, new_cmd_obs=True)


def test_an_invalid_manifest_is_refused(tmp_path):
    directory = _repo(tmp_path, {**ROULADE_MANIFEST, "obs_len": 51})
    with pytest.raises(AssertionError, match="obs_len"):
        ip.resolve_policy_argument(ip.POLICY_SLOTS_BY_NAME["roulade"], str(directory), default_duration=2.0, new_cmd_obs=True)


def test_kind_mismatch_warns_but_still_loads(tmp_path, capsys):
    directory = _repo(tmp_path, pm.build_manifest(name="hold", kind="perpetual", description="held pose", unwind_s=1.5))
    resolved = ip.resolve_policy_argument(ip.POLICY_SLOTS_BY_NAME["roulade"], str(directory), default_duration=2.0, new_cmd_obs=True)
    assert resolved.duration_s == 2.0
    assert "WARNING" in capsys.readouterr().out
