import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 0. 環境與資料準備
# ==========================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qscgrn import *

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")

if not os.path.exists(file_path):
    print(f"錯誤：找不到 {file_path}")
    sys.exit()

df_all = pd.read_csv(file_path)

# 核心 6 基因
target_genes = ['MYB', 'MAFB', 'PPARG', 'EGR2', 'STAT1', 'BCL6']
columns_to_keep = target_genes + ['h']
df_qgrn_ready = df_all[columns_to_keep]

# 建立輸出資料夾，避免檔案太亂
output_dir = os.path.join(current_dir, "results_timecourse")
os.makedirs(output_dir, exist_ok=True)

# Kouno 論文中的真實時間點
time_points = [0, 1, 6, 12, 24, 48, 72, 96]
threshold_val = 5.0

print(f"準備分析完整時間序列: {time_points}")
print("-" * 50)

# ==========================================
# 1. 時間序列批次運算迴圈
# ==========================================
for i in range(len(time_points) - 1):
    t_start = time_points[i]
    t_end = time_points[i+1]
    
    print(f"\n🚀 開始訓練階段: {t_start}h -> {t_end}h")
    
    # 提取起點與終點資料
    df_t0 = df_qgrn_ready[df_qgrn_ready['h'] == t_start].drop(columns=['h'])
    df_t1 = df_qgrn_ready[df_qgrn_ready['h'] == t_end].drop(columns=['h'])
    
    ncells_t1, ngenes = df_t1.shape
    genes = target_genes
    
    # 計算基礎狀態與目標分佈
    activation_t0 = qsc_activation_ratios(df_t0, threshold=threshold_val)
    theta = theta_init(genes, activation_ratios=activation_t0)
    p_obs_t1 = qsc_distribution(df_t1, threshold=threshold_val)
    edges = edges_init(genes)
    
    # 訓練模型 (凍結起點狀態)
    qgrn = model(ncells=ncells_t1, genes=genes, theta=theta, edges=edges, 
                 p_obs=p_obs_t1, epochs=150, save_theta=True, train_encoder=False)
    qgrn.train()
    
    # 匯出權重 CSV
    csv_filename = os.path.join(output_dir, f"causal_theta_{t_start}h_to_{t_end}h.csv")
    qgrn.export_training_theta(csv_filename)
    
    # ==========================================
    # 2. 繪製並儲存網路圖
    # ==========================================
    trained_theta = qgrn.theta
    G = nx.DiGraph()
    G.add_nodes_from(genes)
    
    edge_threshold = 0.01 
    
    for u, v in edges:
        weight_uv = trained_theta[(u, v)]
        if abs(weight_uv) > edge_threshold:
            color = 'green' if weight_uv > 0 else 'red'
            width = abs(weight_uv) * 5 
            G.add_edge(u, v, weight=weight_uv, color=color, width=width)
            
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#A0CBE2', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    edges_to_draw = G.edges(data=True)
    if edges_to_draw: # 確保有連線才畫箭頭，避免報錯
        edge_colors = [edge[2]['color'] for edge in edges_to_draw]
        edge_widths = [edge[2]['width'] for edge in edges_to_draw]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors,
                               width=edge_widths, arrows=True, arrowstyle='-|>',  
                               arrowsize=35, node_size=3000, connectionstyle='arc3,rad=0.15') 
                               
    plt.title(f"Directed Causal Network ({t_start}h -> {t_end}h)", fontsize=18, fontweight='bold')
    plt.axis('off')
    
    svg_filename = os.path.join(output_dir, f"causal_network_{t_start}h_to_{t_end}h.svg")
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    plt.close() # 關閉畫布釋放記憶體
    
    print(f"✅ {t_start}h -> {t_end}h 網路圖已儲存！")

print("\n🎉 完整時間序列分析完成！請檢查 results_timecourse 資料夾。")