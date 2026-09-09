class Parameter:
    value: float
    min: float
    max: float
    optimize: bool
    def __init__(self, value: float, min: float, max: float, optimize: bool = True) -> None: ...
