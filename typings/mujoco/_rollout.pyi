import numpy
import numpy.typing
import typing

class Rollout:
    def __init__(self, nthread: typing.SupportsInt | typing.SupportsIndex) -> None: ...
    def rollout(self, model: list, data: list, nstep: typing.SupportsInt | typing.SupportsIndex, control_spec: typing.SupportsInt | typing.SupportsIndex, state0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], warmstart0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., control: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., state: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., sensordata: typing.Annotated[numpy.typing.ArrayLike, numpy.float64] | None = ..., chunk_size: typing.SupportsInt | typing.SupportsIndex | None = ...) -> None: ...
