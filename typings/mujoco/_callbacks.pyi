# native extension module: pyright cannot introspect it, so fall back to Any
from typing import Any

def __getattr__(name: str) -> Any: ...
