from _typeshed import Incomplete
from bam import simulate as simulate
from bam.logs import Logs as Logs
from bam.model import load_model as load_model

arg_parser: Incomplete
args: Incomplete
logs: Incomplete
params_dir: Incomplete
param_files: Incomplete

def compute_mae(model, log: dict) -> float: ...
def compute_maes_mjlab(param_file, all_logs: list) -> list: ...

results: Incomplete
model: Incomplete
label: Incomplete
maes: Incomplete
mean_mae: Incomplete
std_mae: Incomplete
labels: Incomplete
per_log: Incomplete
means: Incomplete
medians: Incomplete
order: Incomplete
means = means[order]
medians = medians[order]
fig: Incomplete
ax: Incomplete
positions: Incomplete
bp: Incomplete
