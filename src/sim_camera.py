# Wire format matches what mediad captures: its own TCP port carrying length-prefixed raw UYVY
# frames (4-byte little-endian length, then bytes). Rendering costs 12.2 ms per frame against 0.3 ms
# to step four ducks' physics, so cameras are opt-in per duck.

from __future__ import annotations

import socket
import socketserver
import struct
import threading

import mujoco
import numpy as np

# MuJoCo caps offscreen rendering at the model's <global offwidth/offheight>, 640x480 by default.
WIDTH = 640
HEIGHT = 360

# Half the sensor's 30 Hz: a rendered frame costs 12 ms.
FPS = 15

# BT.601, matching duck_detect's sampler on the robot.
_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def to_uyvy(rgb: np.ndarray) -> bytes:
    # Chroma is averaged across each pair, not dropped: taking the left pixel's chroma puts a
    # half-pixel colour shift into every frame, which a detector trained on a real camera will see.
    frame = rgb.astype(np.float32)
    luma = frame @ _Y
    chroma_u = frame @ _U + 128.0
    chroma_v = frame @ _V + 128.0

    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


class Camera:
    # The renderer is not thread-safe and is expensive to build, so only the frame loop touches it.

    def __init__(self, model: mujoco.MjModel, name: str, width: int = WIDTH, height: int = HEIGHT):
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert self.camera >= 0, f"model has no camera {name!r}"

        # The MJCF head camera faces backwards (view direction -x where forward is +x). Turn about
        # the camera's RIGHT axis, not its up axis: both fix the direction, only this one keeps the
        # image the same way up and leaves frames rotated a quarter turn as the real mount is, so
        # `mediad --rotate 90` stays correct in sim. Patched here to leave the RL-owned MJCF alone.
        turn = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degrees about x, scalar-first
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, model.cam_quat[self.camera], turn)
        model.cam_quat[self.camera] = fixed
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.width = width
        self.height = height
        self.latest: bytes | None = None
        self.lock = threading.Lock()

    def render(self, world) -> None:
        # update_scene reads all of MjData on the step thread while sensor reads run on socket
        # threads. Unlocked, a ToF read caught a site orientation mid-write and got a zero-length ray
        # direction, which makes mj_ray abort the process. Hold the lock for the scene copy (~1 ms)
        # and release it for the render (~12 ms, touches no shared state).
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest


class FrameHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        camera: Camera = self.server.camera
        fps: int = self.server.fps
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        period = 1.0 / max(1, fps)
        import time

        next_frame = time.perf_counter()
        try:
            while True:
                frame = camera.frame()
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                next_frame += period
                slack = next_frame - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_frame = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


class FrameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
