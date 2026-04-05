import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 0. 強制讀取本地修改版的 qscgrn 套件
# ==========================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qscgrn import *

# ==========================================
# 1. 準備資料與讀取檔案
# ==========================================
print("正在準備資料...")
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")

if not os.path.exists(file_path):
    print(f"錯誤：找不到 {file_path} 檔案。請確認 rna.csv 是否在 dataset 資料夾中。")
    sys.exit()

df_all = pd.read_csv(file_path)
target_genes = ['MYB', 'MAFB', 'PPARG', 'EGR2', 'STAT1', 'BCL6']
columns_to_keep = target_genes + ['h']
df_qgrn_ready = df_all[columns_to_keep]

# 提取資料 (注意：這裡已經移除了 .T，維持 "細胞 x 基因" 的正確形狀)
df_t0 = df_qgrn_ready[df_qgrn_ready['h'] == 0].drop(columns=['h'])
df_t1 = df_qgrn_ready[df_qgrn_ready['h'] == 1].drop(columns=['h'])

ncells_t1, ngenes = df_t1.shape
genes = target_genes

print(f"t0 細胞數: {df_t0.shape[0]}, t1 細胞數: {ncells_t1}, 訓練基因數: {ngenes}")

# ==========================================
# 2. 核心設定與訓練模型
# ==========================================
threshold_val = 5.0

print("\n正在計算 0h 的基礎狀態 (t0)...")
activation_t0 = qsc_activation_ratios(df_t0, threshold=threshold_val)
theta = theta_init(genes, activation_ratios=activation_t0)

print("正在計算 1h 的目標分佈 (t1)...")
p_obs_t1 = qsc_distribution(df_t1, threshold=threshold_val)
edges = edges_init(genes)

print("\n開始訓練單向量子因果調控網路...")
# 關鍵參數：train_encoder=False (凍結 t0 狀態，只訓練連線)
qgrn = model(ncells=ncells_t1,
             genes=genes,
             theta=theta,
             edges=edges,
             p_obs=p_obs_t1,
             epochs=150,
             save_theta=True,
             train_encoder=False)

qgrn.train()
print("\n訓練完成！")

# 匯出權重數據 CSV 備份
qgrn.export_training_theta("causal_theta_0h_to_1h_6genes.csv")

# ==========================================
# 3. 繪製明確單向因果網路圖 (DAG)
# ==========================================
print("\n正在繪製明確單向因果網路圖...")

trained_theta = qgrn.theta
G = nx.DiGraph() # 宣告為有向圖 (Directed Graph)
G.add_nodes_from(genes)

# 只顯示權重絕對值大於 0.01 的顯著連線
edge_threshold = 0.01 

for u, v in edges:
    weight_uv = trained_theta[(u, v)]

    if abs(weight_uv) > edge_threshold:
        # 正調控設為綠色，負調控設為紅色
        color = 'green' if weight_uv > 0 else 'red'
        # 權重絕對值越大，線條越粗
        width = abs(weight_uv) * 5 

        G.add_edge(u, v, weight=weight_uv, color=color, width=width)

pos = nx.circular_layout(G)
plt.figure(figsize=(10, 8))

# 繪製節點與標籤
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#A0CBE2', edgecolors='black')
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# 繪製明確的單向箭頭
edges_to_draw = G.edges(data=True)
edge_colors = [edge[2]['color'] for edge in edges_to_draw]
edge_widths = [edge[2]['width'] for edge in edges_to_draw]

nx.draw_networkx_edges(G, pos,
                       edgelist=edges_to_draw,
                       edge_color=edge_colors,
                       width=edge_widths,
                       arrows=True,       # 開啟箭頭
                       arrowstyle='-|>',  # 箭頭樣式
                       arrowsize=35,      # 箭頭大小
                       node_size=3000,
                       connectionstyle='arc3,rad=0.15') # 增加弧度避免雙向連線重疊

plt.title("Directed Causal Regulatory Network (0h -> 1h)", fontsize=18, fontweight='bold')
plt.axis('off')

# 儲存與顯示
filename = "causal_network_0h_to_1h_directed.svg"
plt.savefig(filename, format='svg', bbox_inches='tight')
print(f"單向有向網路圖已儲存為: {filename}")
print("所有任務順利完成！")