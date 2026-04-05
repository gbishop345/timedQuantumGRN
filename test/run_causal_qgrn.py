import os
import numpy as np
import pandas as pd
from qscgrn import *

# ==========================================
# 1. Load Data with Relative Path
# ==========================================
# 獲取當前執行位置 (test/)，並組合出 dataset/rna.csv 的路徑
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "../dataset/rna.csv")

print(f"Reading data from: {file_path} ...")
df_all = pd.read_csv(file_path)

# Manually specify 6 core genes
target_genes = ['MYB', 'MAFB', 'PPARG', 'EGR2', 'STAT1', 'BCL6']
print(f"\n6 core genes for this training: {target_genes}")

# Reconstruct DataFrame
columns_to_keep = target_genes + ['h']
df_qgrn_ready = df_all[columns_to_keep]

# ==========================================
# 2. Extract 0h (t0) and 1h (t1) Data
# ==========================================
df_t0 = df_qgrn_ready[df_qgrn_ready['h'] == 0].drop(columns=['h']).T
df_t1 = df_qgrn_ready[df_qgrn_ready['h'] == 1].drop(columns=['h']).T

ncells_t1, ngenes = df_t1.shape
genes = target_genes
print(f"t0 cell count: {df_t0.shape[1]}, t1 cell count: {df_t1.shape[1]}, Training gene count: {ngenes}")

# ==========================================
# 3. Core Settings & Threshold Correction
# ==========================================
threshold_val = 5.0  

print("\nCalculating basal state for 0h (t0)...")
activation_t0 = qsc_activation_ratios(df_t0, threshold=threshold_val)
theta = theta_init(genes, activation_ratios=activation_t0)

print("Calculating target distribution for 1h (t1)...")
p_obs_t1 = qsc_distribution(df_t1, threshold=threshold_val)
edges = edges_init(genes)

# ==========================================
# 4. Build and Train QGRN Model
# ==========================================
print("\nStarting training of the directed quantum causal regulatory network...")
# train_encoder=False freezes t0 state
qgrn = model(ncells=ncells_t1, 
             genes=genes, 
             theta=theta, 
             edges=edges, 
             p_obs=p_obs_t1, 
             epochs=150,           
             save_theta=True,
             train_encoder=False)  

qgrn.train()

# ==========================================
# 5. Output and Visualize Results
# ==========================================
print("\nTraining complete! Exporting charts...")
p_out = qgrn.p_out.reshape(2**ngenes,)

# 輸出檔案會自動儲存在你執行程式的 test/ 資料夾中
comparison_hist(ngenes, p_obs_t1, p_out, limit=0.01, ymax=0.15, 
                filename="causal_comparison_0h_to_1h_6genes.svg")

draw_network(genes, edges, qgrn.theta, filename="causal_network_0h_to_1h_6genes.svg")

qgrn.export_training_theta("causal_theta_0h_to_1h_6genes.csv")

print("All results saved successfully in the 'test' directory!")