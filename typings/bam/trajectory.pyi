def cubic_interpolate(keyframes: list, t: float) -> float: ...

class Trajectory:
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

class LiftAndDrop(Trajectory):
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

class SinusTimeSquare(Trajectory):
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

class UpAndDown(Trajectory):
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

class SinSin(Trajectory):
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

class Nothing(Trajectory):
    duration: float
    def __call__(self, t: float) -> tuple[float, bool]: ...

trajectories: dict[str, Trajectory]
