import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pickle
import torch


def plot_loss(graph, title=None, name=None):
    """
    Plot loss curve, including its course during the optimization.
    """
    node1, node2 = list[graph]
    N = len(graph[node1][node2].keys())-1

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.gnuplot(np.linspace(1,0,N)))
    if title is not None:
        plt.title(f"{name}.{graph_name}")
    plt.gcf().set_size_inches(8, 6)
    for i in range(0, N):
        x_dists = np.concatenate(([0], np.cumsum(np.repeat([graph[node1][node2][i+1]["lengths"].numpy() / 10], 10))))
        x_dists_norm = x_dists / x_dists[-1]
        plt.plot(x_dists_norm, graph[node1][node2][i+1]["dense_train_loss"])#, ".")
    plt.xlabel("Position along the path", fontsize=18)
    plt.ylabel("Loss", fontsize=19)
    # add markers for where optimized points lie
    ylim = plt.gca().get_ylim()
    plt.scatter(x_dists_norm[::10], [ylim[0]] * len(x_dists_norm[::10]), c="0", marker=".")
    # change y tick labels from 20000 to 20k
    if ylim[0] > 10000:
        ylabels = [f"{int(i/1000)}k" for i in plt.gca().get_yticks()]
        plt.gca().set_yticklabels(ylabels)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    #print(plt.gcf().get_size_inches(), plt.gcf().dpi, plt.gcf().get_size_inches()*plt.gcf().dpi)

    print("highest pivot:", np.argmax(graph[node1][node2][N]["dense_train_loss"]).item(), "    loss at highest:", max(graph[node1][node2][N]["dense_train_loss"]).item())
    print("left end:", graph[node1][node2][N]["dense_train_loss"][0].item(), "        right end:", graph[node1][node2][N]["dense_train_loss"][-1].item())

    if name is not None:
        plt.savefig(f"figures/loss/{name}.{graph_name}.loss.png", format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(f"figures/loss/{name}.{graph_name}.loss.svg", format="svg", bbox_inches="tight")
        plt.savefig(f"figures/loss/{name}.{graph_name}.loss.pdf", format="pdf", bbox_inches="tight")


def plot_embedding(embedding, labels, cmap="nipy_spectral", label_names=None):
    """
    Plot an embedding, including a color with optional label names.
    """
    num_classes = len(np.unique(labels))
    ax.scatter(*embedding.T, c=labels, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=num_classes), s=0.2, alpha=0.5, edgecolors="none")
    if label_names is None:
        label_names = np.arange(num_classes)
    cbar = plt.colorbar(sc, ax=axs, ticks=np.arange(len(label_names))+0.5, aspect=80, pad=0.05)
    cbar.ax.set_xticklabels(label_names)
    cbar.solids.set_alpha(1)
    cbar.outline.set_visible(False)