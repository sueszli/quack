from mujoco._callbacks import *
from mujoco._constants import *
from mujoco._enums import *
from mujoco._errors import *
from mujoco._functions import *
from mujoco._specs import *
from mujoco._structs import *
from mujoco._render import *
from mujoco.rendering.classic.gl_context import *
from _typeshed import Incomplete
from mujoco import _specs
from mujoco.rendering.classic.renderer import Renderer as Renderer
from typing import Any, IO, Sequence
from typing_extensions import TypeAlias

proc_translated: Incomplete
is_rosetta: Incomplete
MJTNUM_DTYPE: Incomplete
MjStruct: TypeAlias

def to_zip(spec: _specs.MjSpec, file: str | IO[bytes]) -> None: ...
def from_zip(file: str | IO[bytes]) -> _specs.MjSpec: ...

class _MjBindModel:
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...
    def __setattr__(self, key: str, value: Any): ...

class _MjBindData:
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...
    def __setattr__(self, key: str, value: Any): ...

HEADERS_DIR: Incomplete
PLUGINS_DIR: Incomplete
PLUGIN_HANDLES: Incomplete
__version__: Incomplete
