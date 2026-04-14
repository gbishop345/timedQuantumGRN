import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 0. 環境與路徑初始化
# ==========================================
# 強制載入包含 Time Lag 修改的本地 qscgrn 套件
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qscgrn import *

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "../dataset/rna.csv")

if not os.path.exists(file_path):
    print(f"錯誤：找不到數據檔案 {file_path}")
    sys.exit()

# 讀取數據並定義 6 個核心研究基因
df_all = pd.read_csv(file_path)
target_genes = ['MYB', 'MAFB', 'PPARG', 'EGR2', 'STAT1', 'BCL6']
columns_to_keep = target_genes + ['h']
df_qgrn_ready = df_all[columns_to_keep]

# 建立自動化輸出資料夾
output_dir = os.path.join(current_dir, "results_timecourse")
os.makedirs(output_dir, exist_ok=True)

# 定義 Kouno 數據集的非線性時間序列
time_points = [0, 1, 6, 12, 24, 48, 72, 96]
threshold_val = 5.0

print(f"🚀 開始執行時間序列量子因果分析 (包含 Time Lag 引擎)")
print("-" * 60)

# ==========================================
# 1. 跨時間點自動化訓練迴圈
# ==========================================
for i in range(len(time_points) - 1):
    t_start = time_points[i]
    t_end = time_points[i+1]
    
    # 計算時間增量 (Delta T) 作為量子矩陣演化的次方數
    delta_t = int(t_end - t_start)
    
    print(f"\n分析階段: {t_start}h -> {t_end}h")
    print(f">> 自動偵測到時間延遲 (Time Lag): {delta_t} 小時")
    
    # 提取起點與終點細胞數據
    df_t0 = df_qgrn_ready[df_qgrn_ready['h'] == t_start].drop(columns=['h'])
    df_t1 = df_qgrn_ready[df_qgrn_ready['h'] == t_end].drop(columns=['h'])
    
    ncells_t1, ngenes = df_t1.shape
    genes = target_genes
    
    # 初始化量子狀態與邊界條件
    activation_t0 = qsc_activation_ratios(df_t0, threshold=threshold_val)
    theta = theta_init(genes, activation_ratios=activation_t0)
    p_obs_t1 = qsc_distribution(df_t1, threshold=threshold_val)
    edges = edges_init(genes)
    
    # 建立 QGRN 模型
    qgrn = model(ncells=ncells_t1, genes=genes, theta=theta, edges=edges, 
                 p_obs=p_obs_t1, epochs=150, save_theta=True, train_encoder=False)
    
    # 【注入靈魂】：將計算出的 Delta T 傳入底層矩陣次方引擎
    qgrn.time_lag = delta_t
    
    # 執行梯度下降訓練
    qgrn.train()
    
    # 匯出權重數據 CSV
    csv_filename = os.path.join(output_dir, f"causal_theta_{t_start}h_to_{t_end}h.csv")
    qgrn.export_training_theta(csv_filename)
    
    # ==========================================
    # 2. 繪製單向有向因果網路圖 (DAG)
    # ==========================================
    trained_theta = qgrn.theta
    G = nx.DiGraph()
    G.add_nodes_from(genes)
    
    edge_threshold = 0.01 
    
    for u, v in edges:
        weight_uv = trained_theta[(u, v)]
        if abs(weight_uv) > edge_threshold:
            # 正調控=綠色, 負調控=紅色
            color = 'green' if weight_uv > 0 else 'red'
            width = abs(weight_uv) * 5 
            G.add_edge(u, v, weight=weight_uv, color=color, width=width)
            
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    
    # 繪製節點標籤
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#A0CBE2', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    
    # 繪製具備正確偏移的箭頭
    edges_to_draw = G.edges(data=True)
    if edges_to_draw:
        edge_colors = [edge[2]['color'] for edge in edges_to_draw]
        edge_widths = [edge[2]['width'] for edge in edges_to_draw]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors,
                               width=edge_widths, arrows=True, arrowstyle='-|>',  
                               arrowsize=35, node_size=3000, # node_size 確保箭頭不被遮擋
                               connectionstyle='arc3,rad=0.15') 
                               
    plt.title(f"Causal Network ({t_start}h -> {t_end}h, Lag={delta_t})", fontsize=18, fontweight='bold')
    plt.axis('off')
    
    svg_filename = os.path.join(output_dir, f"causal_network_{t_start}h_to_{t_end}h.svg")
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"✅ 階段性分析完成，圖檔與權重已儲存至 results_timecourse 資料夾。")

print("\n🎉 恭喜！完整時間序列的量子模擬任務已順利結束。")