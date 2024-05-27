import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
import torch


def plot_loss(graph, title=None, name=None):
    """
    Plot loss curve, including its course during the optimization.

    Black points below the curve indicate the positions of pivots that
    were directly optimized by autoneb.

    Parameters
    ----------
    graph : networkx.MultiGraph
        The graph created and optimized by autoneb.
    title : str, optional
        Optional title to be shown above the loss curve.
    name : str, optional
        If a name is given, the plot will be saved as .png, .svg and .pdf to
        the path "figures/loss/{name}.loss.pdf .
    """
    node1, node2 = list(graph)
    N = len(graph[node1][node2].keys())-1

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.gnuplot(np.linspace(1,0,N)))
    if title is not None:
        plt.title(f"{title}")
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

    # print important loss values
    print(f'highest pivot: {np.argmax(graph[node1][node2][N]["dense_train_loss"]).item():13}   highest loss: {max(graph[node1][node2][N]["dense_train_loss"]).item()}')
    print(f'left end: {graph[node1][node2][N]["dense_train_loss"][0].item():18}   right end: {graph[node1][node2][N]["dense_train_loss"][-1].item()}')

    if name is not None:
        plt.savefig(f"figures/loss/{name}.loss.png", format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(f"figures/loss/{name}.loss.svg", format="svg", bbox_inches="tight")
        plt.savefig(f"figures/loss/{name}.loss.pdf", format="pdf", bbox_inches="tight")
        print(f"Plot has been saved as figures/loss/{name}.loss.png, .svg, .pdf")



def to_array(embs, dtype=np.float64):
    """Converts iterable of embeddings to 3D numpy array of given dtype.

    Parameters
    ----------
    embs : np.ndarray, torch.Tensor, list of np.array or torch.Tensor
        The iterable of embeddings to be converted. If only one embedding is passed (2D),
        an extra dimension will be added.
    dtype : str or np.dtype, default=np.float64
    """
    if isinstance(embs, list):
        if isinstance(embs[0], np.ndarray): # list of np.ndarrays
            embs = np.array(embs)
        elif isinstance(embs[0], torch.Tensor): # list of torch.Tensors
            embs = np.array([emb.numpy() for emb in embs])
        else: # list of unknown datatype
            print("Unknown embedding datatype", type(embs[0]))
    elif isinstance(embs, np.ndarray):
        pass
    elif isinstance(embs, torch.Tensor):
        embs = embs.numpy()
    else:
        print("\"embs\" needs to be list, np.ndarray or torch.Tensor, but was", type(embs))

    if len(embs.shape) <= 2: # single embedding was given
        embs = embs[None]

    assert len(embs.shape) == 3
    assert isinstance(embs[0], np.ndarray)
    
    return embs.astype(np.float64)



def plot_embedding(embedding, y_data, labels=None, cmap="nipy_spectral", scatter_kwargs={}):
    """Plot an embedding, including a colorbar with optional label names.

    Parameters
    ----------
    embedding : np.ndarray or torch.Tensor
        The embedding to be plotted. Needs to be of size (N, 2) with N the number of data points.
    y_data : np.ndarray or torch.Tensor
        1D array containing the class label for each data point of an embedding.
        Needs to be of length N, same as the embedding.
    labels : list or np.ndarray, optional
        The names of each class as strings.
    cmap : matplotlib colormap or str, default="nipy_spectral"
        Either a colormap of choice is passed directly, or
        the name of a matplotlib colormap is passed.
    scatter_kwargs : dict, default={}
        Keyword arguments that will be passed on to the plt.scatter function.
        Default values are s=0.6, alpha=0.5 but can be overwritten by values in scatter_kwargs.
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.numpy()
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.numpy()

    num_classes = len(np.unique(y_data))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap, num_classes)

    if labels is None:
        labels = sorted(np.unique(y_data))

    # set default kwargs for scatter plot
    scatter_kwargs.setdefault("s", 0.6)
    scatter_kwargs.setdefault("alpha", 0.5)
    scatter_kwargs.setdefault("edgecolors", "none")

    # set axis properties
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().set_box_aspect(aspect=1)

    # plot the embedding
    sc = plt.scatter(*embedding.T, c=y_data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=num_classes), **scatter_kwargs)

    # create colorbar
    cbar = plt.colorbar(sc, ax=plt.gca(), ticks=np.arange(len(labels))+0.5, aspect=80, pad=0.05)
    cbar.ax.set_yticklabels(labels)
    cbar.solids.set_alpha(1)
    cbar.outline.set_visible(False)



def plot_embeddings(embeddings, y_data, labels=None, cmap="nipy_spectral", columns=3, scale=1.0, forced_size=None, scatter_kwargs={}):
    """Plot multiple embeddings in a grid, including a shared colorbar.
    
    Parameters
    ----------
    embeddings : np.ndarray, torch.Tensor, list of np.array or torch.Tensor
        The embeddings which will be plotted in a grid. Each embedding needs
        to be of size (N, 2) with N the number of data points.
    y_data : np.ndarray or torch.Tensor
        1D array containing the class label for each data point of an embedding.
        Needs to be of same length N as each embedding.
    labels : list or np.ndarray, optional
        The names of each class as strings.
    cmap : matplotlib colormap or str, default="nipy_spectral"
        Either a colormap of choice is passed directly, or
        the name of a matplotlib colormap is passed.
    columns : int, default=3
        Number of embeddings to be plotted side by side in each row.
    scale : float, default=1.0
        Scale to set the overall figure size. Default means a size of (5, 6) inches for each embedding.
    forced_size : tuple, default=None
        Fixed total figure size in inches. If forced_size is set, scale parameter will be ignored.
    scatter_kwargs : dict, default={}
        Keyword arguments that will be passed on to the plt.scatter function.
        Default values are s=0.2, alpha=0.5 but can be overwritten by values in scatter_kwargs.
    """
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.numpy()
    
    num_classes = len(np.unique(y_data))

    # if no label names are given, just use numbers in ascending order
    if labels is None:
        labels = sorted(np.unique(y_data))

    # get matplotlib colormap or use the specified one
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap, num_classes)

    # set figure size and layout
    rows = int(np.ceil(len(embeddings) / columns))
    fig, axs = plt.subplots(rows, columns, sharex=True, sharey=True)
    if forced_size is not None:
        fig.set_size_inches(forced_size)
        if scale != 1:
            print("Parameter \"scale\" was overridden by \"force_size\".")
    else:
        fig.set_size_inches((scale*5*columns, scale*6*rows))
        
    # set default kwargs for scatter plot
    scatter_kwargs.setdefault("s", 0.2)
    scatter_kwargs.setdefault("alpha", 0.5)

    # plot each embedding
    for i, ax in enumerate(axs.flatten()):
        ax.set_box_aspect(aspect=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # leave axis blank if no embedding left
        if i < len(embeddings):
            sc = ax.scatter(*embeddings[i].T, c=y_data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=num_classes), **scatter_kwargs)
            ax.set_title(i)

    # create colorbar
    cbar = plt.colorbar(sc, ax=axs, orientation="horizontal", ticks=np.arange(len(labels))+0.5, aspect=40, pad=0.05)
    cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(rotation=90)
    cbar.solids.set_alpha(1)
    cbar.outline.set_visible(False)