import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 그래프 함수
def graph(df, idx, col, figsize=(5, 5)):
    # 폰트 설정
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False

    corr = df[idx].corr()
    target = col
    G = nx.Graph()

    # 노드/엣지 추가
    for c in corr.columns:
        if c == target:
            continue
        val = corr.loc[target, c]
        if abs(val) >= 0.7:
            if abs(val) >= 0.9:
                width = 8  # 아주 두껍게
            else:
                width = abs(val) * 10  # 기본 비례
            color = 'red' if val > 0 else 'blue'
            G.add_node(c)
            G.add_edge(target, c, weight=width, color=color)

    # 레이아웃
    pos = nx.spring_layout(G, seed=42, weight='weight', k=0.3)
    edges = G.edges()

    # 시각화
    colors = [G[u][v]['color'] for u, v in edges]
    widths = [G[u][v]['weight'] for u, v in edges]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='Malgun Gothic')
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, alpha=0.5)

    # 패딩 조정
    x_vals = [v[0] for v in pos.values()]
    y_vals = [v[1] for v in pos.values()]
    x_pad = (max(x_vals) - min(x_vals)) * 0.6
    y_pad = (max(y_vals) - min(y_vals)) * 0.2
    plt.xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
    plt.ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)

    # 테두리 제거
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()
