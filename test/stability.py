"""
Edge stability for the **same** time-course setup as ``run_timecourse_qgrn.py``.

For each consecutive ``(t, t+1)`` in ``time_points`` (one QGRN per transition, like
``run_timestep_sequence``), trains the model ``N_STABILITY_RUNS`` times and counts
how often each directed edge is active. Writes a single CSV under
``results_timecourse``; does not write network SVGs or theta traces.

Run from repo root: ``python test/stability.py``
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd

# Repo root on path (same pattern as run_timecourse_qgrn.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qscgrn import train_timestep_grn
from qscgrn.qcircuit.utils import edges_init
from qscgrn.utils import GRN_EDGE_DISPLAY_THRESHOLD_RAD

# ---------------------------------------------------------------------------
# Mirror test/run_timecourse_qgrn.py so stability is per the same timecourse.
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")
output_dir = os.path.join(current_dir, "results_timecourse")
time_points = [0, 1, 6, 12, 24, 48, 72, 96]
target_genes = ["MYB", "MAFB", "PPARG", "EGR2", "STAT1", "BCL6"]
time_col = "h"
threshold_val = "median"
epochs = 150
train_encoder = False
# Same edge visibility rule as run_timestep_sequence default
edge_plot_threshold = GRN_EDGE_DISPLAY_THRESHOLD_RAD

N_STABILITY_RUNS = 5
STABILITY_CSV_NAME = "grn_edge_stability.csv"


def _present_edges(
    qgrn, edge_threshold: float
) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for u, v in qgrn.edges:
        if abs(float(qgrn.theta.loc[(u, v)])) > edge_threshold:
            out.append((u, v))
    return out


def run_stability_analysis(
    df: pd.DataFrame,
    genes: Sequence[str],
    time_points_arg: Sequence[Union[int, float]],
    *,
    time_col_arg: str = time_col,
    threshold=threshold_val,
    epochs_arg: int = epochs,
    n_runs: int = N_STABILITY_RUNS,
    edge_threshold: float = edge_plot_threshold,
    train_encoder_arg: bool = train_encoder,
) -> pd.DataFrame:
    """
    Same transition grid as ``run_timestep_sequence``: one fit per
    ``(time_points[i], time_points[i+1])``, repeated ``n_runs`` times.
    """
    gene_list = list(genes)
    all_edges = edges_init(gene_list)
    pairs = list(zip(time_points_arg[:-1], time_points_arg[1:]))

    counts: Dict[Tuple[Union[int, float], Union[int, float], str, str], int] = (
        defaultdict(int)
    )

    for _ in range(n_runs):
        for t_start, t_end in pairs:
            df_a = df[df[time_col_arg] == t_start][list(genes)]
            df_b = df[df[time_col_arg] == t_end][list(genes)]
            if df_a.empty or df_b.empty:
                continue
            qgrn = train_timestep_grn(
                df,
                gene_list,
                t_start,
                t_end,
                time_col=time_col_arg,
                threshold=threshold,
                epochs=epochs_arg,
                train_encoder=train_encoder_arg,
                save_theta=False,
                drop_zero=True,
            )
            for u, v in _present_edges(qgrn, edge_threshold):
                counts[(t_start, t_end, u, v)] += 1

    rows = []
    for t_start, t_end in pairs:
        df_a = df[df[time_col_arg] == t_start][list(genes)]
        df_b = df[df[time_col_arg] == t_end][list(genes)]
        skipped = df_a.empty or df_b.empty
        for u, v in all_edges:
            c = counts.get((t_start, t_end, u, v), 0)
            rows.append(
                {
                    "t_start": t_start,
                    "t_end": t_end,
                    "time_transition": f"{t_start}h_to_{t_end}h",
                    "control": u,
                    "target": v,
                    "edge": f"{u}-{v}",
                    "n_present": c,
                    "n_runs": n_runs,
                    "present_fraction": c / n_runs,
                    "unanimous": c in (0, n_runs),
                    "pair_skipped": skipped,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    if not os.path.exists(file_path):
        print(f"Error: data file not found: {file_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, STABILITY_CSV_NAME)

    df_all = pd.read_csv(file_path)
    df_qgrn_ready = df_all[target_genes + [time_col]]

    df_out = run_stability_analysis(
        df_qgrn_ready,
        target_genes,
        time_points,
        time_col_arg=time_col,
        threshold=threshold_val,
        epochs_arg=epochs,
        n_runs=N_STABILITY_RUNS,
        edge_threshold=edge_plot_threshold,
        train_encoder_arg=train_encoder,
    )

    lines = [
        "# GRN edge stability (same time grid as run_timecourse_qgrn.py / run_timestep_sequence)",
        f"# {N_STABILITY_RUNS} independent fits per non-skipped time transition",
        f"# Edge active if |theta| > {edge_plot_threshold} (run_timestep_sequence edge_plot_threshold)",
        "# unanimous=True: edge on in all runs or off in all runs",
    ]
    for (t0, t1), g in df_out.groupby(["t_start", "t_end"], sort=False):
        sub = g[~g["pair_skipped"]]
        if sub.empty:
            lines.append(f"# {t0}h_to_{t1}h: SKIPPED (empty slice, same as run_timestep_sequence)")
            continue
        n_edge = len(sub)
        n_unanimous = int(sub["unanimous"].sum())
        n_unstable = n_edge - n_unanimous
        lines.append(
            f"# {t0}h_to_{t1}h: edges_unanimous={n_unanimous}/{n_edge} "
            f"edges_varying_across_runs={n_unstable}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        df_out.to_csv(f, index=False)

    print(f"Wrote {out_path} ({len(df_out)} rows).")
    print(f"(Timecourse config matches {os.path.join(current_dir, 'run_timecourse_qgrn.py')})")


if __name__ == "__main__":
    main()
