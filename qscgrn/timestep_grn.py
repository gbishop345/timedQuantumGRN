"""
One-step (t -> t+1) quantum GRN on a fixed n-qubit register.

For each consecutive pair (t_start, t_end) in a time grid, fit a separate
model: encoder parameters reflect marginals at t_start, optimization matches
the empirical bitstring histogram at t_end (train_encoder=False freezes the
t_start preparation; only regulation edges are learned).
"""
import os
from typing import Any, Dict, List, Sequence, Union

import pandas as pd

from .optimizer import model
from .qcircuit import edges_init, theta_init
from .utils import (
    GRN_EDGE_DISPLAY_THRESHOLD_RAD,
    ThresholdSpec,
    info_print,
    qsc_activation_ratios,
    qsc_distribution,
)

__all__ = [
    "train_timestep_grn",
    "run_timestep_sequence",
    "save_final_theta_csv",
    "export_causal_network_svg",
]

# Edges with |θ| below this are omitted from the “no display threshold” plot
# (float noise only).
_EDGE_ABS_EPS = 1e-12


def train_timestep_grn(
    df: pd.DataFrame,
    genes: Sequence[str],
    t_start: Union[int, float],
    t_end: Union[int, float],
    *,
    time_col: str = "h",
    threshold: ThresholdSpec = 0.0,
    epochs: int = 150,
    train_encoder: bool = False,
    save_theta: bool = True,
    drop_zero: bool = True,
    learning_rate: float = 0.1,
    method: str = "kl-divergence",
) -> model:
    """
    Fit a single QGRN mapping time t_start -> t_end.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``genes`` columns and ``time_col``.
    genes : sequence of str
        Gene names (column order defines qubit order).
    t_start, t_end :
        Values of ``time_col`` for source and target time slices.
    time_col : str
        Name of the time / stage column (default ``h`` for hours).
    threshold : float, str, ndarray, or Series
        Binarization cutoff(s). Use ``"median"`` or ``"mean"`` for per-gene
        cutoffs computed **inside each time slice** (``t_start`` for encoder,
        ``t_end`` for ``p_obs``). A vector or Series sets explicit cutoffs per gene.
    epochs, train_encoder, save_theta, drop_zero, learning_rate, method
        Passed through to ``model`` / training. Regulation-edge L1 with
        proximal soft-thresholding is always applied inside ``model.train``.

    Returns
    -------
    model
        Trained ``model`` instance (final ``theta`` on the instance).
    """
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(f"Missing gene columns: {missing}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column {time_col!r}")

    df_a = df[df[time_col] == t_start][list(genes)]
    df_b = df[df[time_col] == t_end][list(genes)]

    if df_a.empty:
        raise ValueError(f"No rows for {time_col}={t_start}")
    if df_b.empty:
        raise ValueError(f"No rows for {time_col}={t_end}")

    ncells = int(df_b.shape[0])
    gene_list = list(genes)

    activation = qsc_activation_ratios(df_a, threshold=threshold)
    theta = theta_init(gene_list, activation_ratios=activation)
    p_obs = qsc_distribution(df_b, threshold=threshold, drop_zero=drop_zero)
    edges = edges_init(gene_list)

    qgrn = model(
        ncells=ncells,
        genes=gene_list,
        theta=theta,
        edges=edges,
        p_obs=p_obs,
        epochs=epochs,
        save_theta=save_theta,
        train_encoder=train_encoder,
        drop_zero=drop_zero,
        learning_rate=learning_rate,
        method=method,
    )
    qgrn.train()
    return qgrn


def save_final_theta_csv(qgrn: model, path: str) -> None:
    """Write final optimized ``theta`` as a one-row CSV (columns = gene pairs)."""
    cols = [f"{a}-{b}" for a, b in qgrn.theta.index]
    row = pd.DataFrame([qgrn.theta.values], columns=cols)
    row.to_csv(path, index=False)


def export_causal_network_svg(
    qgrn: model,
    path: str,
    *,
    edge_min_abs: float,
    title: str,
) -> None:
    """
    Save a directed network sketch: edges with ``abs(theta) > edge_min_abs``.

    Use ``edge_min_abs=GRN_EDGE_DISPLAY_THRESHOLD_RAD`` for the usual display
    cutoff, or ``edge_min_abs=_EDGE_ABS_EPS`` (or ``0.0``) to show every
    numerically non-zero regulation edge.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(qgrn.genes)
    for u, v in qgrn.edges:
        w = float(qgrn.theta[(u, v)])
        if abs(w) > edge_min_abs:
            color = "green" if w > 0 else "red"
            width = max(abs(w) * 5.0, 0.2)
            G.add_edge(u, v, weight=w, color=color, width=width)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3000,
        node_color="#A0CBE2",
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold")
    edges_draw = list(G.edges(data=True))
    if edges_draw:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges_draw,
            edge_color=[e[2]["color"] for e in edges_draw],
            width=[e[2]["width"] for e in edges_draw],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=35,
            node_size=3000,
            connectionstyle="arc3,rad=0.15",
        )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close()


def run_timestep_sequence(
    df: pd.DataFrame,
    genes: Sequence[str],
    time_points: Sequence[Union[int, float]],
    *,
    time_col: str = "h",
    threshold: ThresholdSpec = 5.0,
    output_dir: str = ".",
    epochs: int = 150,
    train_encoder: bool = False,
    save_theta_trace: bool = True,
    edge_plot_threshold: float = GRN_EDGE_DISPLAY_THRESHOLD_RAD,
    skip_on_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Train one GRN per consecutive pair ``(time_points[i], time_points[i+1])``.

    For each pair, exports:
      - ``causal_theta_{t_start}_to_{t_end}_trace.csv`` (training trace, if
        ``save_theta_trace``),
      - ``theta_final_{t_start}_to_{t_end}.csv`` (final parameters),
      - ``causal_network_{t_start}_to_{t_end}.svg`` (with ``edge_plot_threshold``),
      - ``causal_network_{t_start}_to_{t_end}_all_nonzero.svg`` (no display cutoff,
        only hides ``|θ|`` at numerical zero).

    Returns
    -------
    list of dict
        Each dict has ``t_start``, ``t_end``, ``model``, ``paths`` (paths dict),
        and optionally ``error`` if the pair was skipped.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "run_timestep_sequence requires matplotlib and networkx "
            "for network plots."
        ) from e

    results: List[Dict[str, Any]] = []
    pairs = list(zip(time_points[:-1], time_points[1:]))

    for t_start, t_end in pairs:
        # File tags match the manuscript-style hour labels (e.g. 0h_to_1h).
        tag = f"{t_start}h_to_{t_end}h"
        paths = {
            "theta_final": os.path.join(output_dir, f"theta_final_{tag}.csv"),
            "theta_trace": os.path.join(output_dir, f"causal_theta_{tag}_trace.csv"),
            "network_svg": os.path.join(output_dir, f"causal_network_{tag}.svg"),
            "network_svg_all_nonzero": os.path.join(
                output_dir, f"causal_network_{tag}_all_nonzero.svg"
            ),
        }

        df_a = df[df[time_col] == t_start][list(genes)]
        df_b = df[df[time_col] == t_end][list(genes)]

        if df_a.empty or df_b.empty:
            msg = (
                f"Skipping pair ({t_start}, {t_end}): empty slice "
                f"(rows t_start={df_a.shape[0]}, t_end={df_b.shape[0]})."
            )
            if skip_on_empty:
                info_print(msg, level="W")
                results.append(
                    {
                        "t_start": t_start,
                        "t_end": t_end,
                        "model": None,
                        "paths": paths,
                        "skipped": True,
                    }
                )
                continue
            raise ValueError(msg.replace("Skipping pair ", ""))

        qgrn = train_timestep_grn(
            df,
            genes,
            t_start,
            t_end,
            time_col=time_col,
            threshold=threshold,
            epochs=epochs,
            train_encoder=train_encoder,
            save_theta=save_theta_trace,
            drop_zero=True,
        )

        save_final_theta_csv(qgrn, paths["theta_final"])

        if save_theta_trace:
            qgrn.export_training_theta(paths["theta_trace"])
        else:
            paths["theta_trace"] = None

        delta = t_end - t_start if isinstance(t_end, (int, float)) else ""
        export_causal_network_svg(
            qgrn,
            paths["network_svg"],
            edge_min_abs=edge_plot_threshold,
            title=(
                f"QGRN ({t_start}→{t_end}; Δ={delta})  "
                f"|θ| > {edge_plot_threshold:g} rad"
            ),
        )
        export_causal_network_svg(
            qgrn,
            paths["network_svg_all_nonzero"],
            edge_min_abs=_EDGE_ABS_EPS,
            title=(
                f"QGRN ({t_start}→{t_end}; Δ={delta})  "
                "all |θ| > 0 (no display threshold)"
            ),
        )

        results.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "model": qgrn,
                "paths": paths,
                "skipped": False,
            }
        )

    return results
