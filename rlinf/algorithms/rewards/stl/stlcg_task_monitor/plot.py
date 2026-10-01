"""Two-panel robustness plot for one episode.

Top: the task's actual atomic-predicate margins vs step t.
Bottom: suffix (backward) rho(phi, s[t:]) under THREE min/max aggregation methods:
  - true (exact hard max/min) — crimson
  - logsumexp (smooth upper bound) — purple
  - softmax (smooth, biased low) — teal
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import numpy as np


def episode_ids(path: str) -> tuple[int, int]:
    import h5py
    tid = eid = -1
    try:
        with h5py.File(path, "r") as f:
            tid = int(f.attrs.get("task_id", -1))
            eid = int(f.attrs.get("episode_idx", -1))
    except Exception:
        pass
    if tid < 0 or eid < 0:
        m = re.search(r"task(\d+)_ep(\d+)", os.path.basename(path))
        if m:
            tid = int(m.group(1)) if tid < 0 else tid
            eid = int(m.group(2)) if eid < 0 else eid
    return tid, eid


def plot_result(res, out_path: str, title: Optional[str] = None) -> None:
    """Plot the ONLINE PREFIX trace (ρ(φ, s[0:t+1]) — the live verdict that climbs)."""
    trace = res.online_trace if res.online_trace is not None else res.suffix_trace
    _plot_core(res, {"true": trace}, out_path, title)


def plot_three_methods(res, suffix_traces: Dict[str, np.ndarray], out_path: str,
                       title: Optional[str] = None) -> None:
    """Plot with three aggregation methods overlaid on the bottom panel."""
    _plot_core(res, suffix_traces, out_path, title)


def _plot_core(res, suffix_traces: Dict[str, np.ndarray], out_path: str,
               title: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(res.T)
    n_atoms = len(res.atoms)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [1.0, 1.15]})

    # ---- top: atoms ----
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_atoms, 1)))
    for (nm, m), c in zip(res.atoms.items(), colors):
        ax1.plot(t, m, color=c, lw=1.6, label=nm)
        fs = res.atom_steps.get(nm)
        if fs is not None:
            ax1.axvline(fs, color=c, ls=":", lw=0.9, alpha=0.6)
    ax1.set_ylabel("atomic predicate margin\n(>0 satisfied)")
    ax1.set_title(f"Atoms  ({res.template})")
    ax1.legend(loc="best", fontsize=7.5)
    ax1.grid(alpha=0.25)

    # ---- bottom: task rho under multiple aggregation methods ----
    ax2.axhline(0.0, color="k", lw=0.9, alpha=0.6)
    style = {
        "true":      dict(color="crimson", lw=2.2, ls="-"),
        "logsumexp": dict(color="purple",  lw=1.6, ls="-."),
        "softmax":   dict(color="teal",    lw=1.4, ls="--"),
    }
    labels = {
        "true":      r"true (exact $\max/\min$)",
        "logsumexp": r"logsumexp ($\tau$=10)",
        "softmax":   r"softmax ($\tau$=10)",
    }
    for method, trace in suffix_traces.items():
        if trace is None:
            continue
        s = style.get(method, dict(color="gray", lw=1.2))
        lbl = labels.get(method, method)
        final = float(trace[0])
        ax2.plot(t, trace, label=f"{lbl}  ρ={final:+.4f}", **s)

    sat = "SATISFIED" if res.satisfied else "VIOLATED"
    methods_tag = "+".join(suffix_traces.keys())
    ax2.set_title(f"Task $\\phi$ — online $\\rho(\\phi,s[0:t{{+}}1])$  —  final: {sat}")
    ax2.set_xlabel("step t")
    ax2.set_ylabel("task robustness margin")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_xlim(0, res.T - 1)

    fig.suptitle(title or res.task_description, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"plot -> {out_path}")
