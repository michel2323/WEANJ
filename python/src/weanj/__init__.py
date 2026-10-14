"""Shared helpers for the WEANJ example notebooks (JAX column).

Mirrors the Julia module `WEANJ` (julia/src/WEANJ.jl) so all three implementations
read the same inputs and write results in the same format.
"""

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

# Do not grab 75% of the GPU up front; other processes (and Julia notebooks) share it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "ROOT", "data_path", "load_inputs", "time_median", "write_results",
    "save_gradient", "fd_check", "gpu_busy_mib", "IMPLS",
    "compare_gradients", "load_results", "scaling_plot",
]


def data_path(example: str) -> Path:
    """Path of the shared input file for `example` (e.g. "01-particles")."""
    return ROOT / "data" / f"{example}.npz"


def load_inputs(example: str) -> dict[str, np.ndarray]:
    """Load the shared inputs written by the Julia side."""
    with np.load(data_path(example)) as f:
        return {k: f[k] for k in f.files}


def time_median(f, *, sync=lambda out: None, warmup: int = 2, samples: int = 10) -> float:
    """Median wall time of `f()` in seconds after `warmup` untimed calls.

    `sync` receives the return value of `f()` inside the timed region; pass
    `jax.block_until_ready` for asynchronous JAX dispatch.
    """
    for _ in range(warmup):
        sync(f())
    ts = []
    for _ in range(samples):
        t0 = time.perf_counter()
        sync(f())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def write_results(example: str, impl: str, device: str, records: list[dict]) -> Path:
    """Write benchmarks/results/<example>_<impl>_<device>.json.

    Each record has keys dtype, N, forward_s, grad_s, compile_s, peak_mem_bytes
    (None for unavailable values).
    """
    d = ROOT / "benchmarks" / "results"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{example}_{impl}_{device}.json"
    path.write_text(json.dumps(
        {"example": example, "impl": impl, "device": device, "records": records},
        indent=2))
    return path


def save_gradient(example: str, impl: str, g) -> Path:
    """Store the Float64 correctness-check gradient for cross-implementation comparison."""
    d = ROOT / "benchmarks" / "grads"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{example}_{impl}.npz"
    np.savez(path, grad=np.asarray(g, dtype=np.float64))
    return path


def fd_check(f, x, g, *, n: int = 5, h: float = 1e-6, rtol: float = 1e-5, seed: int = 0) -> float:
    """Compare `n` random gradient components against central finite differences.

    Returns the maximum relative error; raises if it exceeds `rtol`.
    """
    xh = np.array(x, dtype=np.float64)
    gh = np.asarray(g)
    flat, gflat = xh.reshape(-1), gh.reshape(-1)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in rng.integers(0, flat.size, n):
        xi = flat[i]
        flat[i] = xi + h; fp = float(f(xh))
        flat[i] = xi - h; fm = float(f(xh))
        flat[i] = xi
        fd = (fp - fm) / (2 * h)
        worst = max(worst, abs(fd - gflat[i]) / max(abs(fd), abs(gflat[i]), 1e-12))
    if worst > rtol:
        raise AssertionError(f"finite-difference check failed: max rel. error {worst} > {rtol}")
    return worst


def gpu_busy_mib() -> int:
    """GPU memory currently in use by all processes, in MiB (via nvidia-smi)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


IMPLS = ("enzyme", "reactant", "jax")


def compare_gradients(example: str, impls=IMPLS, rtol: float = 1e-8):
    """Check that all implementations saved the same Float64 gradient.

    Returns a pandas DataFrame of pairwise max relative differences against the first
    implementation; raises if any exceeds `rtol`.
    """
    import pandas as pd

    grads = {}
    for impl in impls:
        with np.load(ROOT / "benchmarks" / "grads" / f"{example}_{impl}.npz") as f:
            grads[impl] = f["grad"]
    ref_name, ref = impls[0], grads[impls[0]]
    rows = []
    for impl in impls:
        g = grads[impl]
        if g.shape != ref.shape:
            raise AssertionError(f"{impl} gradient shape {g.shape} != {ref_name} {ref.shape}")
        rel = float(np.max(np.abs(g - ref)) / max(np.max(np.abs(ref)), 1e-300))
        rows.append({"implementation": impl, "‖g‖₂": float(np.linalg.norm(g)),
                     f"max rel. diff vs {ref_name}": rel})
    df = pd.DataFrame(rows)
    bad = df[df.iloc[:, 2] > rtol]
    if len(bad):
        raise AssertionError(f"gradients disagree beyond rtol={rtol}:\n{bad}")
    return df


def load_results(example: str):
    """All timing records for `example` as one pandas DataFrame."""
    import pandas as pd

    rows = []
    for path in sorted((ROOT / "benchmarks" / "results").glob(f"{example}_*.json")):
        d = json.loads(path.read_text())
        for r in d["records"]:
            rows.append({"impl": d["impl"], "device": d["device"], **r})
    return pd.DataFrame(rows)


def scaling_plot(example: str, metric: str = "grad_s", title: str | None = None,
                 query: str | None = None, x: str = "N", x_label: str | None = None):
    """Log-log plot of `metric` against `x`, one line per (implementation, variant).

    `query` filters the records first (pandas query syntax, e.g. "K == 100").
    """
    import plotly.express as px

    df = load_results(example)
    if query:
        df = df.query(query)
    label = df["impl"]
    if "variant" in df:  # e.g. JAX "scan" vs "red-black"
        label = label + df["variant"].map(lambda v: f" / {v}" if isinstance(v, str) else "")
    df = df.assign(implementation=label + " (" + df["device"] + ")")
    fig = px.line(df.sort_values(x), x=x, y=metric, color="implementation",
                  markers=True, log_x=True, log_y=True, title=title,
                  labels={"grad_s": "gradient time [s]", "forward_s": "forward time [s]",
                          "peak_mem_bytes": "memory [bytes]",
                          **({x: x_label} if x_label else {})})
    return fig
