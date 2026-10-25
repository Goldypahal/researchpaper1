"""
Benchmark Suite Package.
Exposes Dyck-k, Hidden FSM, and Legacy Parity benchmarks.
"""

from benchmarks.dyck import (
    get_dyck_benchmark_splits,
    verify_sequence_validity,
    is_open_bracket,
    is_close_bracket,
    OPEN_BRACKETS,
    CLOSE_BRACKETS,
    BRACKET_MATCH
)
from benchmarks.fsm import (
    HiddenFSM,
    get_fsm_benchmark_splits
)
from benchmarks.parity_legacy import (
    get_parity_benchmark_splits
)

from benchmarks.abs_position import (
    get_abs_position_benchmark_splits
)

def load_benchmark(task: str, **kwargs):
    task = task.lower()
    if task == "dyck":
        return get_dyck_benchmark_splits(**kwargs)
    elif task == "fsm":
        return get_fsm_benchmark_splits(**kwargs)
    elif task in ("abs_position", "abs_pos"):
        return get_abs_position_benchmark_splits(**kwargs)
    elif task in ("parity", "parity_legacy"):
        return get_parity_benchmark_splits(**kwargs)
    else:
        raise ValueError(f"Unknown benchmark task '{task}'. Choose from ['dyck', 'fsm', 'abs_position', 'parity']")
