import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx

from utils import uniquify


def plot_loss(graph, title=None, save_as=None, nodes=None, figsize=(8, 6)):
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
    save_as : str, optional
        If a name is given, the plot will be saved as .png, .svg and .pdf to
        the path "figures/loss/{save_as}.loss.pdf
    nodes : tuple, optional
        Tuple containing the names of two nodes. The plot will show the
        loss along the connection between the two.
    figsize : tuple, default=(8, 6)
        The total size of the figure in inches.
    """
    # use first two nodes only
    if nodes is None:
        node1, node2 = list(graph)[:2]
    else:
        node1, node2 = nodes[:2]

    # get number of optimization cycles, excluding first dummy cycle
    N = len(graph[node1][node2].keys())-1

    # define colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.gnuplot(np.linspace(1,0,N)))

    if title is not None:
        plt.title(f"{title}")
    
    #set figure size
    plt.gcf().set_size_inches(*figsize)

    for i in range(0, N):
        x_dists = np.concatenate(([0], np.cumsum(np.repeat([graph[node1][node2][i+1]["lengths"].numpy() / 10], 10))))
        x_dists_norm = x_dists / x_dists[-1]
        plt.plot(x_dists_norm, graph[node1][node2][i+1]["dense_train_loss"])#, ".")
    plt.xlabel("Position along the path", fontsize=18)
    plt.ylabel("Loss", fontsize=19)

    # add markers for where optimized points lie
    ylim = plt.gca().get_ylim()
    plt.scatter(x_dists_norm[::10], [ylim[0]] * len(x_dists_norm[::10]), c="0", marker=".")

    # change y tick label format from e.g. 20000 to 20k
    if ylim[0] > 10000:
        ylabels = [f"{int(i/1000)}k" for i in plt.gca().get_yticks()]
        plt.gca().set_yticks(plt.gca().get_yticks())
        plt.gca().set_yticklabels(ylabels)

    # formatting
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # print important loss values
    print(f'highest pivot: {np.argmax(graph[node1][node2][N]["dense_train_loss"]).item():13}   highest loss: {max(graph[node1][node2][N]["dense_train_loss"]).item()}')
    print(f'left end: {graph[node1][node2][N]["dense_train_loss"][0].item():18}   right end: {graph[node1][node2][N]["dense_train_loss"][-1].item()}')

    # save figure
    if save_as is not None:
        filepath_png = uniquify(f"figures/loss/{save_as}.loss.png")
        filepath_svg = uniquify(f"figures/loss/{save_as}.loss.svg")
        filepath_pdf = uniquify(f"figures/loss/{save_as}.loss.pdf")

        plt.savefig(filepath_png, format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(filepath_svg, format="svg", bbox_inches="tight")
        plt.savefig(filepath_pdf, format="pdf", bbox_inches="tight")
        print(f"Plot has been saved as {filepath_png}, .svg, .pdf")


def plot_embedding(embedding, y_data, labels=None, cmap="nipy_spectral", scatter_kwargs={}, figsize=(8, 6), save_as=None):
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
    figsize : tuple, default=(8, 6)
        The total size of the figure in inches.
    save_as : str, optional
        If a name is given, the plot will be saved as .png, .svg and .pdf to
        the path "figures/embs/{save_as}.pdf
    """
    # convert inputs to np.ndarray
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

    # set figure size
    plt.gcf().set_size_inches(*figsize)

    # plot the embedding
    sc = plt.scatter(*embedding.T, c=y_data, cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=num_classes), **scatter_kwargs)

    # create colorbar
    cbar = plt.colorbar(sc, ax=plt.gca(), ticks=np.arange(len(labels))+0.5, aspect=80, pad=0.05)
    cbar.ax.set_yticklabels(labels)
    cbar.solids.set_alpha(1)
    cbar.outline.set_visible(False)

    # save figure
    if save_as is not None:
        filepath_png = uniquify(f"figures/embs/{save_as}.png")
        filepath_svg = uniquify(f"figures/embs/{save_as}.svg")
        filepath_pdf = uniquify(f"figures/embs/{save_as}.pdf")

        plt.savefig(filepath_png, format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(filepath_svg, format="svg", bbox_inches="tight")
        plt.savefig(filepath_pdf, format="pdf", bbox_inches="tight")
        print(f"Plot has been saved as {filepath_png}, .svg, .pdf")


def plot_embeddings(embeddings, y_data, labels=None, cmap="nipy_spectral", columns=3, scale=1.0, forced_size=None, scatter_kwargs={}, save_as=None):
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
    save_as : str, optional
        If a name is given, the plot will be saved as .png, .svg and .pdf to
        the path "figures/embs/{save_as}.pdf
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
    cbar = plt.colorbar(sc, ax=axs, orientation="horizontal", ticks=np.arange(len(labels))+0.5, aspect=80, pad=0.10)
    cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(rotation=90)
    cbar.solids.set_alpha(1)
    cbar.outline.set_visible(False)
        
    # save figure
    if save_as is not None:
        filepath_png = uniquify(f"figures/embs/{save_as}.png")
        filepath_svg = uniquify(f"figures/embs/{save_as}.svg")
        filepath_pdf = uniquify(f"figures/embs/{save_as}.pdf")

        plt.savefig(filepath_png, format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(filepath_svg, format="svg", bbox_inches="tight")
        plt.savefig(filepath_pdf, format="pdf", bbox_inches="tight")
        print(f"Plot has been saved as {filepath_png}, .svg, .pdf")


def plot_pointwise(embedding, y_data=None, loss_inst=None, cmap="nipy_spectral", cmap_loss="plasma", plot_std=True, plot_att=True, plot_rep=True, save_as=None, att_clip=(0, None), rep_clip=(0, None), scatter_kwargs={}, ax_size=(6, 5), cbar_shrink=1.0, title_size=16, label_size=14):
    """Plot each point's individual contribution to the loss.

    Up to three plots are created:
    The standard embedding visualization with colors,
    the embedding with each point's color corresponding to its attractive loss,
    the embedding with each point's color corresponding to its repulsive loss.

    Parameters
    ----------
    embedding : np.ndarray or torch.Tensor
        The embedding for which to plot the pointwise losses.
    y_data : np.ndarray or torch.Tensor, optional
        1D array containing the class label for each data point of an embedding.
        Needs to be of length N, same as the embedding.
        Can only be omitted if plot_std is set to False.
    loss_inst : UMAP_loss, optional
        An instance of the loss class used for creating the embedding.
        Can only be omitted if both plot_att and plot_rep are set to False.
    cmap : matplotlib colormap or str, default="nipy_spectral"
        The colormap for coloring the standard embedding according to class.
        Either a colormap of choice is passed directly, or
        the name of a matplotlib colormap is passed.
    cmap_loss : matplotlib colormap or str, default="plasma"
        The colormap for the pointwise loss plots. Use att_clip and
        rep_clip parameters for normalization.
        Either a colormap of choice is passed directly, or
        the name of a matplotlib colormap is passed.
    plot_std : bool, default=True
        Whether to plot the standard embedding with class colors.
    plot_att : bool, default=True
        Whether to plot the pointwise attractive loss.
    plot_rep : bool, default=True
        Whether to plot the pointwise repulsive loss.
    save_as : str, optional
        If a name is given, the plot will be saved as .png, .svg and .pdf to
        the path "figures/pointwise/{save_as}.pdf
    att_clip : tuple, default=(0, None)
        Specifies (min, max) values for the attractive loss. Clipping the values
        is helpful to minimize the effect of outliers on color normalization.
    rep_clip : tuple, default=(0, None)
        Specifies (min, max) values for the repulsive loss. Clipping the values
        is helpful to minimize the effect of outliers on color normalization.
    scatter_kwargs : dict, default={}
        Keyword arguments that will be passed on to the plt.scatter function.
        Default values are s=0.2, alpha=0.5 but can be overwritten by values in scatter_kwargs.
    ax_size : tuple, default=(6, 5)
        Size of one subplot in inches, including the colorbar.
    cbar_shrink = float, default=1.0
        Scaling factor for the colorbars.
    title_size : int, default=16
        Font size for the axis titles.
    label_size : int, default=14
        Font size for the axis and colorbar labels.
    """    
    if plot_std:
        if y_data is None:
            print("y_data need to be passed if plot_std==True")
        else:
            num_classes = len(np.unique(y_data))
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap, num_classes)

    if plot_att or plot_rep:
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).to(device=f"cuda:{loss_inst.cuda_device}", dtype=loss_inst.dtype)
        elif isinstance(embedding, torch.Tensor):
            embedding = embedding.to(device=f"cuda:{loss_inst.cuda_device}", dtype=loss_inst.dtype)
        else:
            print(f"embedding needs to be np.ndarray or torch.Tensor but was {type(embedding)}")

    if plot_att:
        # get indivdual attractive loss contributions
        c_att_weights = loss_inst(embedding, pointwise=True)[1].flatten().cpu().numpy()
        c_att = -np.bincount(loss_inst.sparse_high.row, weights=c_att_weights)
    if plot_rep:
        # get individual repulsive loss contributions
        c_rep = -loss_inst(embedding, pointwise=True)[2].flatten().cpu().numpy()

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()

    # initialize figure
    num_plots = np.sum([1 if flag else 0 for flag in [plot_std, plot_att, plot_rep]])
    fig, [axs] = plt.subplots(1, num_plots, sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(ax_size[0]*num_plots, ax_size[1])
    fig.tight_layout()

    # set default kwargs for scatter plot
    scatter_kwargs.setdefault("s", 0.2)
    scatter_kwargs.setdefault("alpha", 0.5)

    ax_idx = 0

    # plot standard embedding with colors according to class
    if plot_std:
        axs[ax_idx].set_aspect('equal', adjustable='box')
        sc0 = axs[ax_idx].scatter(*embedding.T, c=y_data, cmap=cmap, edgecolors='none', **scatter_kwargs)
        axs[ax_idx].set_title("UMAP Embedding", fontsize=title_size)
        axs[ax_idx].tick_params(axis='both', which='major', labelsize=label_size)
        cbar0 = fig.colorbar(sc0, ax=axs[ax_idx], shrink=cbar_shrink)
        cbar0.outline.set_visible(False)
        cbar0.solids.set_alpha(0.0)
        cbar0.set_ticks([])

        ax_idx += 1

    # plot embedding with colors according to pointwise attractive loss
    if plot_att:
        axs[ax_idx].set_aspect('equal', adjustable='box')
        sc1 = axs[ax_idx].scatter(*embedding.T, c=np.clip(c_att, *att_clip), norm=matplotlib.colors.LogNorm(), cmap=cmap_loss, edgecolors='none', **scatter_kwargs)
        axs[ax_idx].set_title("Attractive Loss (sum over pairs)", fontsize=title_size)
        axs[ax_idx].tick_params(axis='both', which='major', labelsize=label_size)
        cbar1 = fig.colorbar(sc1, ax=axs[ax_idx], shrink=cbar_shrink)
        cbar1.outline.set_visible(False)
        cbar1.ax.tick_params(labelsize=label_size)

        ax_idx += 1

    # plot embedding with colors according to pointwise repulsive loss
    if plot_rep:
        axs[ax_idx].set_aspect('equal', adjustable='box')
        sc2 = axs[ax_idx].scatter(*embedding.T, c=np.clip(c_rep, *rep_clip), norm=matplotlib.colors.LogNorm(), cmap=cmap_loss, edgecolors='none', **scatter_kwargs)
        axs[ax_idx].set_title("Repulsive Loss", fontsize=title_size)
        axs[ax_idx].tick_params(axis='both', which='major', labelsize=label_size)
        cbar2 = fig.colorbar(sc2, ax=axs[ax_idx], shrink=cbar_shrink)
        cbar2.outline.set_visible(False)
        cbar2.ax.tick_params(labelsize=label_size)

        ax_idx += 1

    [ax.spines["top"].set_visible(False) for ax in axs]
    [ax.spines["right"].set_visible(False) for ax in axs]

    # save figure
    if save_as is not None:
        filepath_png = uniquify(f"figures/pointwise/{save_as}.png")
        filepath_svg = uniquify(f"figures/pointwise/{save_as}.svg")
        filepath_pdf = uniquify(f"figures/pointwise/{save_as}.pdf")

        plt.savefig(filepath_png, format="png", bbox_inches="tight", dpi=1600/max(*plt.gcf().get_size_inches()))
        plt.savefig(filepath_svg, format="svg", bbox_inches="tight")
        plt.savefig(filepath_pdf, format="pdf", bbox_inches="tight")
        print(f"Plot has been saved as {filepath_png}, .svg, .pdf")