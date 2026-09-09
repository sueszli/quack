from dynamixel_sdk import *
from _typeshed import Incomplete

ADDR_TORQUE_ENABLE: int
ADDR_P_GAIN: int
ADDR_GOAL_POSITION: int
ADDR_PRESENT_POSITION: int
ADDR_PRESENT_SPEED: int
ADDR_PRESENT_LOAD: int
ADDR_PRESENT_VOLTAGE: int
ADDR_PRESENT_TEMPERATURE: int
XL320_ADDR_CW_ANGLE_LIMIT: int
XL320_ADDR_CCW_ANGLE_LIMIT: int
XL320_ADDR_CONTROL_MODE: int
XL320_ADDR_MAX_TORQUE: int
XL320_ADDR_TORQUE_ENABLE: int
XL320_ADDR_LED: int
XL320_ADDR_D_GAIN: int
XL320_ADDR_I_GAIN: int
XL320_ADDR_P_GAIN: int
XL320_ADDR_GOAL_POSITION: int
XL320_ADDR_MOVING_SPEED: int
XL320_ADDR_TORQUE_LIMIT: int
XL320_ADDR_PRESENT_POSITION: int
XL320_ADDR_PRESENT_SPEED: int
XL320_ADDR_PRESENT_LOAD: int
XL320_ADDR_PRESENT_VOLTAGE: int
XL320_ADDR_PRESENT_TEMPERATURE: int
XL320_ADDR_MOVING: int
XL320_ADDR_HW_ERROR_STATUS: int
XL320_RESOLUTION: int
XL320_RANGE_DEG: float
XL320_CENTER: int

class DynamixelActuatorV1:
    id: Incomplete
    portHandler: Incomplete
    packetHandler: Incomplete
    def __init__(self, port: str, id: int = 1) -> None: ...
    def set_p_gain(self, gain: int): ...
    def set_torque(self, enable: bool): ...
    def set_goal_position(self, position: float): ...
    def read_data(self): ...

class DynamixelXL320:
    id: Incomplete
    portHandler: Incomplete
    packetHandler: Incomplete
    def __init__(self, port: str, id: int = 1) -> None: ...
    def set_torque(self, enable: bool): ...
    def set_p_gain(self, gain: int): ...
    def set_i_gain(self, gain: int): ...
    def set_d_gain(self, gain: int): ...
    def set_pid_gains(self, p: int, i: int, d: int): ...
    def set_goal_position(self, position: float): ...
    def set_moving_speed(self, speed_rpm: float): ...
    def set_led(self, color: int): ...
    def read_data(self) -> dict: ...
