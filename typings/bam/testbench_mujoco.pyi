import mujoco
from _typeshed import Incomplete

class Pendulum:
    G: float
    mass: Incomplete
    arm_mass: Incomplete
    length: Incomplete
    def __init__(self, log: dict) -> None: ...
    def inertial_params(self) -> tuple[float, float, float]: ...
    def build_spec(self, name: str = 'pendulum') -> mujoco.MjSpec: ...
