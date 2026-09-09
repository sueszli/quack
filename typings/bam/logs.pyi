from . import message as message
from _typeshed import Incomplete

class Logs:
    directory: str
    json_files: Incomplete
    logs: Incomplete
    def __init__(self, directory: str) -> None: ...
    def split(self, selector_kp: int) -> Logs: ...
    def make_batch(self) -> dict: ...
