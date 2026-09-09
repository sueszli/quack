from .actuator import Actuator as Actuator
from .actuators import actuators as actuators
from .parameter import Parameter as Parameter
from _typeshed import Incomplete

class Model:
    actuator_name: Incomplete
    name: Incomplete
    title: Incomplete
    load_dependent: bool
    directional: bool
    stribeck: bool
    quadratic: bool
    max_friction_base: float
    max_load_friction: float
    max_viscous_friction: float
    def __init__(self, load_dependent: bool = False, directional: bool = False, stribeck: bool = False, quadratic: bool = False, name: str = None, title: str = '') -> None: ...
    def reset(self) -> None: ...
    actuator: Incomplete
    q_offset: Incomplete
    friction_base: Incomplete
    friction_stribeck: Incomplete
    load_friction_motor: Incomplete
    load_friction_external: Incomplete
    load_friction_base: Incomplete
    load_friction_motor_stribeck: Incomplete
    load_friction_external_stribeck: Incomplete
    load_friction_stribeck: Incomplete
    load_friction_motor_quad: Incomplete
    load_friction_external_quad: Incomplete
    dtheta_stribeck: Incomplete
    alpha: Incomplete
    friction_viscous: Incomplete
    # Created dynamically by DCMotorActuator.initialize() when an actuator is
    # attached, so stubgen cannot see them on the class body.
    kt: Parameter
    R: Parameter
    def set_actuator(self, actuator: Actuator) -> None: ...
    def compute_frictions(self, motor_torque: float, external_torque: float, dtheta: float) -> tuple: ...
    def get_parameters(self) -> dict: ...
    def get_parameter_values(self) -> dict: ...
    def load_parameters(self, json_file: str) -> list: ...
    def load_parameters_from_dict(self, data: dict) -> list: ...

class DummyModel(Model):
    def __init__(self) -> None: ...

models: Incomplete

def load_model(json_file: str = None, *, motor_name: str = None, model: str = None) -> Model: ...
def load_model_from_dict(data: dict) -> Model: ...
