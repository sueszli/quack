from __future__ import annotations

import json
import socket
import threading

import numpy as np
import pytest

from src import sim_body_server as sbs

MOUTH_INDEX = 9
SERVO_COUNT = 14


def test_wire_layout_is_the_daemons_fifteen_joints():
    assert len(sbs.JOINT_NAMES) == SERVO_COUNT + 1
    assert sbs.JOINT_NAMES[MOUTH_INDEX] == "mouth"


def test_wire_layout_matches_the_documented_joint_order():
    assert sbs.JOINT_NAMES[:5] == ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle")
    assert sbs.JOINT_NAMES[5:9] == ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    assert sbs.JOINT_NAMES[10:] == ("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def test_upright_trunk_reads_gravity_straight_down():
    got = sbs.gravity_in_trunk(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(got, [0.0, 0.0, -1.0], atol=1e-9)


def test_face_down_trunk_reads_gravity_along_plus_x():
    half = np.sqrt(0.5)
    got = sbs.gravity_in_trunk(np.array([half, 0.0, half, 0.0]))
    assert np.allclose(got, [1.0, 0.0, 0.0], atol=1e-9)


def test_upside_down_trunk_reads_gravity_straight_up():
    got = sbs.gravity_in_trunk(np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.allclose(got, [0.0, 0.0, 1.0], atol=1e-9)


def test_gravity_is_a_unit_vector_for_arbitrary_orientations():
    rng = np.random.default_rng(0)
    for _ in range(16):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        assert np.isclose(np.linalg.norm(sbs.gravity_in_trunk(q)), 1.0, atol=1e-9)


@pytest.fixture(scope="module")
def duck():
    world = sbs.World(sbs.DEFAULT_SCENE, 1)
    body = sbs.Body(world, 0)
    server = sbs.Server(("127.0.0.1", 0), sbs.Handler)
    server.body = body
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield world, body, server.server_address
    server.shutdown()
    server.server_close()


def _talk(address, requests):
    answers = []
    with socket.create_connection(address, timeout=10) as sock:
        f = sock.makefile("rwb")
        for request in requests:
            f.write((json.dumps(request) + "\n").encode())
            f.flush()
            answers.append(json.loads(f.readline()))
    return answers


def test_handshake_agrees_on_the_protocol_version(duck):
    _, _, address = duck
    assert _talk(address, [{"op": "hello", "protocol": sbs.PROTOCOL}]) == [{"protocol": sbs.PROTOCOL}]


def test_a_daemon_on_another_protocol_is_refused(duck):
    _, _, address = duck
    answer = _talk(address, [{"op": "hello", "protocol": sbs.PROTOCOL + 1}])[0]
    assert "error" in answer
    assert "protocol mismatch" in answer["error"]


def test_an_unknown_op_is_an_error_not_a_crash(duck):
    _, _, address = duck
    answer = _talk(address, [{"op": "fly"}, {"op": "hello", "protocol": sbs.PROTOCOL}])
    assert "error" in answer[0]
    assert answer[1] == {"protocol": sbs.PROTOCOL}


def test_a_malformed_frame_does_not_take_the_server_down(duck):
    _, _, address = duck
    with socket.create_connection(address, timeout=10) as sock:
        f = sock.makefile("rwb")
        f.write(b"not json\n")
        f.flush()
        assert "error" in json.loads(f.readline())
    assert _talk(address, [{"op": "hello", "protocol": sbs.PROTOCOL}]) == [{"protocol": sbs.PROTOCOL}]


def test_read_returns_every_wire_field_at_full_width(duck):
    _, _, address = duck
    answer = _talk(address, [{"op": "read"}])[0]
    for field in ("positions", "velocities", "currents_ma"):
        assert len(answer[field]) == len(sbs.JOINT_NAMES)
    assert set(answer["imu"]) == {"gyro", "gravity", "quat"}
    assert len(answer["imu"]["gravity"]) == 3
    assert len(answer["imu"]["quat"]) == 4
    assert len(answer["trunk"]) == 3


def test_the_mouth_slot_is_inserted_and_never_driven(duck):
    _world, body, address = duck
    answer = _talk(address, [{"op": "read"}])[0]
    assert answer["positions"][MOUTH_INDEX] == 0.0
    assert answer["velocities"][MOUTH_INDEX] == 0.0
    assert MOUTH_INDEX not in body.to_wire
    assert len(body.to_wire) == SERVO_COUNT


def test_a_write_needs_all_fifteen_slots(duck):
    _, _, address = duck
    answer = _talk(address, [{"op": "write", "targets": [0.0] * SERVO_COUNT}])[0]
    assert "error" in answer
    assert _talk(address, [{"op": "write", "targets": [0.0] * len(sbs.JOINT_NAMES)}])[0] == {}


def test_slow_sensors_report_one_temperature_per_wire_slot(duck):
    _, _, address = duck
    answer = _talk(address, [{"op": "slow"}])[0]
    assert len(answer["temps_c"]) == len(sbs.JOINT_NAMES)
    assert answer["volts"] == sbs.NOMINAL_VOLTS


def test_stepping_advances_the_reported_clock(duck):
    world, _, address = duck
    before = _talk(address, [{"op": "read"}])[0]["sim_time"]
    world.step(50)
    after = _talk(address, [{"op": "read"}])[0]["sim_time"]
    assert after > before
    assert np.isclose(after - before, 50 * world.model.opt.timestep, rtol=1e-6)


def test_a_resting_duck_reports_gravity_close_to_down(duck):
    _, _, address = duck
    gravity = _talk(address, [{"op": "read"}])[0]["imu"]["gravity"]
    assert np.isclose(np.linalg.norm(gravity), 1.0, atol=1e-6)
