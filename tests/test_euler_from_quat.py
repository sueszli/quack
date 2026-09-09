import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat


def _euler_xyz_inline(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True)


_EDGE_QUATS = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.70710678, 0.0], [0.70710678, 0.0, -0.70710678, 0.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)


def test_matches_inline_expansion_on_edge_poses():
    quat = _normalize(_EDGE_QUATS)
    for got, want in zip(euler_xyz_from_quat(quat), _euler_xyz_inline(quat), strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)


def test_matches_inline_expansion_on_random_orientations():
    torch.manual_seed(0)
    quat = _normalize(torch.randn(20000, 4, dtype=torch.float64))
    for got, want in zip(euler_xyz_from_quat(quat), _euler_xyz_inline(quat), strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)


def test_gimbal_lock_pitch_is_exactly_quarter_turn():
    quat = _normalize(_EDGE_QUATS[1:3])
    _, pitch, _ = euler_xyz_from_quat(quat)
    assert torch.isfinite(pitch).all()
    torch.testing.assert_close(pitch.abs(), torch.full_like(pitch, torch.pi / 2))


def test_float32_agreement():
    torch.manual_seed(1)
    quat = _normalize(torch.randn(20000, 4))
    for got, want in zip(euler_xyz_from_quat(quat), _euler_xyz_inline(quat), strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0)
