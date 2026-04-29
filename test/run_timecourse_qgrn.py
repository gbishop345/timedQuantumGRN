import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qscgrn import run_timestep_sequence

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")

if not os.path.exists(file_path):
    print(f"Error: data file not found: {file_path}")
    sys.exit(1)

df_all = pd.read_csv(file_path)
target_genes = ["MYB", "MAFB", "PPARG", "EGR2", "STAT1", "BCL6"]
df_qgrn_ready = df_all[target_genes + ["h"]]

output_dir = os.path.join(current_dir, "results_timecourse")
time_points = [0, 1, 6, 12, 24, 48, 72, 96]
# Per-gene median within each time slice (encoder uses t, p_obs uses t+1).
# Use a scalar (e.g. 5.0) for one global raw-count cutoff instead.
threshold_val = "median"

print("Training one QGRN per consecutive (t, t+1) pair in time_points.")
print("Each map: encoder from marginals at t, loss vs histogram at t+1.")
print("-" * 60)

results = run_timestep_sequence(
    df_qgrn_ready,
    target_genes,
    time_points,
    time_col="h",
    threshold=threshold_val,
    output_dir=output_dir,
    epochs=150,
    train_encoder=False,
    save_theta_trace=True,
)

n_ok = sum(1 for r in results if not r.get("skipped"))
print(f"Done. Trained {n_ok} / {len(results)} pairs. Outputs in: {output_dir}")
