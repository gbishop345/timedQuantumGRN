import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qscgrn import export_causal_network_svg, save_final_theta_csv, train_timestep_grn
from qscgrn.utils import GRN_EDGE_DISPLAY_THRESHOLD_RAD

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")

if not os.path.exists(file_path):
    print(f"Error: data file not found: {file_path}")
    sys.exit(1)

df_all = pd.read_csv(file_path)
target_genes = ["MYB", "MAFB", "PPARG", "EGR2", "STAT1", "BCL6"]
df_qgrn_ready = df_all[target_genes + ["h"]]

genes = target_genes
threshold_val = "median"  # or 5.0 for a single raw-count cutoff

print("Single pair 0h -> 1h (same one-step map as run_timestep_sequence).")
qgrn = train_timestep_grn(
    df_qgrn_ready,
    genes,
    t_start=0,
    t_end=1,
    time_col="h",
    threshold=threshold_val,
    epochs=150,
    train_encoder=False,
    save_theta=True,
)

save_final_theta_csv(qgrn, "causal_theta_0h_to_1h_final.csv")
qgrn.export_training_theta("causal_theta_0h_to_1h_6genes_trace.csv")

export_causal_network_svg(
    qgrn,
    "causal_network_0h_to_1h_directed.svg",
    edge_min_abs=GRN_EDGE_DISPLAY_THRESHOLD_RAD,
    title=(
        "Directed causal QGRN (0h→1h)  "
        f"|θ| > {GRN_EDGE_DISPLAY_THRESHOLD_RAD:g} rad"
    ),
)
export_causal_network_svg(
    qgrn,
    "causal_network_0h_to_1h_all_nonzero.svg",
    edge_min_abs=1e-12,
    title="Directed causal QGRN (0h→1h)  all |θ| > 0 (no display threshold)",
)
print(
    "Saved: causal_theta CSVs, causal_network_0h_to_1h_directed.svg (with threshold), "
    "causal_network_0h_to_1h_all_nonzero.svg (no display threshold)."
)
