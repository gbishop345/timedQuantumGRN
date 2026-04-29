import numpy as np
from ..utils import info_print


__all__ = ["draw_network"]


def _coordinates_graph(ngenes):
    angle = 2 * np.pi / ngenes
    angles = np.arange(ngenes) * angle
    x = np.cos(angles) + 1
    y = np.sin(angles) + 1
    return x, y


def draw_network(genes, edges, theta, threshold=1, filename=None):
    """
    Draw the network representation of the QuantumGRN model.

    Uses NetworkX and Matplotlib (no igraph / Cairo).

    Parameters
    ----------
    genes : list
        The gene list from the dataset.
    edges : list
        The edges for the QuantumGRN model.
    theta : pd.Series
        The theta values in the QuantumGRN model.
    threshold : float
        The threshold for dropping edges from the network. Details in the
        manuscript.
    filename : str or None
        Path to export the figure (e.g. ``.png`` or ``.svg``). If None, the
        figure is shown with ``plt.show()``.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    info_msg = " and exporting to {file} file.".format(file=filename) \
        if filename is not None else ""
    info_print("Drawing the network representation of the qscGRN model{msg}"
               .format(msg=info_msg))

    theta_e = theta[edges]
    theta_e = theta_e[np.abs(theta_e) > (threshold * np.pi / 180)]
    filtered_edges = theta_e.index.to_list()

    x, y = _coordinates_graph(len(genes))
    pos = {g: (float(x[i]), float(y[i])) for i, g in enumerate(genes)}

    G = nx.DiGraph()
    G.add_nodes_from(genes)
    for edge in filtered_edges:
        w = float(theta_e[edge])
        width = max(0.4, 30.0 * abs(w) / np.pi / 6.0)
        color = "#70AD47" if w >= 0 else "#FF0000"
        G.add_edge(edge[0], edge[1], weight=w, color=color, width=width)

    plt.figure(figsize=(6.5, 6.5))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1800,
        node_color="#DAE3F3",
        edgecolors="#2F528F",
        linewidths=2,
    )
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")
    edgelist = list(G.edges(data=True))
    if edgelist:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            edge_color=[d["color"] for _, _, d in edgelist],
            width=[d["width"] for _, _, d in edgelist],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            node_size=1800,
            connectionstyle="arc3,rad=0.12",
            min_source_margin=18,
            min_target_margin=22,
        )
    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
