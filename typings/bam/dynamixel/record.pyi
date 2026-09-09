from bam.trajectory import *
from .dynamixel import DynamixelActuatorV1 as DynamixelActuatorV1, DynamixelXL320 as DynamixelXL320
from _typeshed import Incomplete

arg_parser: Incomplete
args: Incomplete

def convert_xl330_velocity(raw_signed: float) -> float: ...
def convert_xl330_pwm_to_duty(raw: float) -> float: ...

trajectory: Incomplete
c: Incomplete
ID: int
start: Incomplete
goal_position: Incomplete
torque_enable: Incomplete
dxl: Incomplete
data: Incomplete
t: Incomplete
new_torque_enable: Incomplete
torque_enable = new_torque_enable
t0: Incomplete
entry: Incomplete
t1: Incomplete
return_dt: float
max_variation: Incomplete
date: Incomplete
filename: Incomplete
