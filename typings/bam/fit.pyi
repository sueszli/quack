from . import message as message, simulate as simulate
from .actuators import actuators as actuators
from .logs import Logs as Logs
from .model import Model as Model, load_model as load_model, models as models
from _typeshed import Incomplete

arg_parser: Incomplete
args: Incomplete
params_json_filename: Incomplete
logs: Incomplete
validation_logs: Incomplete
validation_batch: Incomplete
logs_batch: Incomplete

def compute_score(model: Model, log: dict) -> float: ...
def compute_scores(model: Model, compute_logs=None): ...
def make_model() -> Model: ...
def objective(trial): ...

last_log: Incomplete
wandb_run: Incomplete

def monitor(study, trial) -> None: ...

model: Incomplete
study_name: Incomplete
study_url: str
sampler: Incomplete

def optuna_run(enable_monitoring: bool = True) -> None: ...

p: Incomplete
