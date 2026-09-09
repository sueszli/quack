import mujoco
from .model import Model as Model, load_model_from_dict as load_model_from_dict
from .testbench_mujoco import Pendulum as Pendulum
from _typeshed import Incomplete

class MujocoController:
    model: Incomplete
    actuator: Incomplete
    mujoco_model: Incomplete
    mujoco_data: Incomplete
    vin_drop_gain: Incomplete
    vin_min: Incomplete
    dofs: Incomplete
    q_target: Incomplete
    dof_to_q_target: Incomplete
    last_ts: Incomplete
    act_indexes: Incomplete
    joint_indexes: Incomplete
    qpos_indexes: Incomplete
    dof_indexes: Incomplete
    # actuator is np.atleast_1d'd, so a list of joint names is the normal call.
    def __init__(self, model: Model, actuator: str | list[str], mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, vin_drop_gain: float | None = None, vin_min: float | None = None) -> None: ...
    def get_q_target(self, name: str) -> float: ...
    def set_q_target(self, name: str, q_target: float): ...
    def reset(self, qpos) -> None: ...
    def update(self) -> None: ...

class Simulator:
    model: Incomplete
    actuator: Incomplete
    instances: list[tuple]
    t: float
    def __init__(self, model: Model, actuator: str = 'pendulum') -> None: ...
    def reset(self, q: float = 0.0, dq: float = 0.0): ...
    @property
    def q(self): ...
    @property
    def dq(self): ...
    def step(self, goal_position, torque_enable, dt: float): ...
    def rollout_log(self, log: dict, reset_period: float = None): ...

def load_config(path: str, mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, kp: float, vin: float) -> tuple: ...
