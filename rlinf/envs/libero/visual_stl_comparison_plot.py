"""Side-by-side privileged and visual STL trace export.

The plotting layer is deliberately data-only.  It does not evaluate either
predicate family and therefore cannot leak privileged state into visual
scoring.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rlinf.envs.libero.stl_stage_plan import Atom, StagePlan


@dataclass(frozen=True)
class STLComparisonFrame:
    step: int
    privileged_normalized: Mapping[Atom, float]
    visual_normalized: Mapping[Atom, float]
    privileged_score: float
    visual_score: float
    visual_valid: bool
    visual_raw: Mapping[Atom, float]
    visual_atom_valid: Mapping[Atom, bool]
    visual_atom_reasons: Mapping[Atom, str]
    privileged_stage: str = ""
    visual_stage: str = ""
    visual_reason: str = ""


def atom_label(atom: Atom) -> str:
    pred, obj, target = atom
    args = ",".join(value for value in (obj, target) if value is not None)
    return f"{pred}({args})"


def write_stl_comparison_csv(
    frames: Sequence[STLComparisonFrame],
    plan: StagePlan,
    output_path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = tuple(plan.atoms)
    fieldnames = [
        "step", "privileged_score", "visual_score", "visual_valid",
        "privileged_stage", "visual_stage", "visual_reason",
    ]
    for atom in atoms:
        label = atom_label(atom)
        fieldnames.extend((
            f"privileged::{label}",
            f"visual::{label}",
            f"visual_raw::{label}",
            f"visual_valid::{label}",
            f"visual_reason::{label}",
        ))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            row = {
                "step": int(frame.step),
                "privileged_score": float(frame.privileged_score),
                "visual_score": float(frame.visual_score),
                "visual_valid": bool(frame.visual_valid),
                "privileged_stage": str(frame.privileged_stage),
                "visual_stage": str(frame.visual_stage),
                "visual_reason": str(frame.visual_reason),
            }
            for atom in atoms:
                label = atom_label(atom)
                row[f"privileged::{label}"] = float(
                    frame.privileged_normalized.get(atom, float("nan"))
                )
                row[f"visual::{label}"] = float(
                    frame.visual_normalized.get(atom, float("nan"))
                )
                row[f"visual_raw::{label}"] = float(
                    frame.visual_raw.get(atom, float("nan"))
                )
                row[f"visual_valid::{label}"] = bool(
                    frame.visual_atom_valid.get(atom, False)
                )
                row[f"visual_reason::{label}"] = str(
                    frame.visual_atom_reasons.get(atom, "")
                )
            writer.writerow(row)
    return path


def plot_stl_comparison(
    frames: Sequence[STLComparisonFrame],
    plan: StagePlan,
    output_path,
    *,
    title: str,
) -> Path:
    """Draw identical normalized axes: privileged left, visual right."""
    if not frames:
        raise ValueError("at least one comparison frame is required")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = tuple(plan.atoms)
    steps = np.asarray([frame.step for frame in frames], dtype=np.int64)
    colors = plt.get_cmap("tab10")

    figure, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharex=True, sharey=True
    )
    panels = (
        (axes[0], "Privileged STL", "privileged_normalized", "privileged_score"),
        (axes[1], "Visual STL", "visual_normalized", "visual_score"),
    )
    for axis, panel_title, atom_field, score_field in panels:
        for index, atom in enumerate(atoms):
            values = np.asarray([
                getattr(frame, atom_field).get(atom, float("nan"))
                for frame in frames
            ], dtype=np.float64)
            is_pick = atom[0] == "pick"
            axis.plot(
                steps,
                values,
                color=colors(index % 10),
                linewidth=1.7,
                linestyle=":" if is_pick else "-",
                label=atom_label(atom),
            )
            finite = np.flatnonzero(np.isfinite(values) & (values >= 0.0))
            if finite.size:
                axis.axvline(
                    steps[int(finite[0])],
                    color=colors(index % 10),
                    linestyle=":",
                    linewidth=0.8,
                    alpha=0.5,
                )
        score = np.asarray([
            getattr(frame, score_field) for frame in frames
        ], dtype=np.float64)
        axis.plot(
            steps, score, color="black", linewidth=2.4,
            label="overall task STL",
        )
        axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9)
        axis.set_title(panel_title)
        axis.set_xlabel("environment step")
        axis.grid(alpha=0.25)
        axis.set_ylim(-1.05, 0.15)
        axis.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("normalized robustness")
    valid_count = sum(frame.visual_valid for frame in frames)
    figure.suptitle(
        f"{title}\nvisual AGM updates={valid_count}/{len(frames)}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path
