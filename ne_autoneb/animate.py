import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm
from scipy.spatial import procrustes


def graph_to_path(graph, nodes=None, pause=2):
    """Creates a path cycling along the connections between the nodes."""
    if nodes is None:
        nodes = list(graph)

    num_datapoints = graph.nodes[nodes[0]]["coords"].shape[0]

    pause = int(max(0, pause))
    
    print("Node cycle: ", end="")
    Zs = []
    for i in range(len(nodes)):
        node1, node2 = nodes[i], nodes[(i+1) % len(nodes)]
        print(node1, "-> ", end="")
        connection = graph[node1][node2][max(graph[node1][node2])]["path_coords"]
        nearp = torch.reshape(connection[0], (num_datapoints, 2)).clone()
        if procrustes(graph.nodes[node1]["coords"].cpu() - np.average(graph.nodes[node1]["coords"].cpu(), axis=0), nearp - np.average(nearp, axis=0))[2] > procrustes(graph.nodes[node2]["coords"].cpu() - np.average(graph.nodes[node2]["coords"].cpu(), axis=0), nearp - np.average(nearp, axis=0))[2]:
            connection = reversed(connection)
        del nearp
        for path_coord in connection:
            Zs.append(torch.reshape(path_coord, (path_coord.shape[0] // 2, 2)).numpy())
        Zs = Zs + [Zs[-1]] * pause
    print(node2)

    return Zs


def create_animation(graph, labels, nodes=None, interpolate=3, center=(0, 0), lim=50, cmap="nipy_spectral", fps=12.5, pause=2, size=6, scatter_kwargs={}):
    """Animate the connection between two nodes of the graph.
    
    Parameters
    ----------
    graph : networkx.MultiGraph
        The graph created by autoneb containing all nodes and connections.
    labels : np.ndarray or torch.Tensor
        The class label for each data point of an embedding.
    nodes : tuple, optional
        Tuple specifying for which nodes the connection should be animated.
    interpolate : int, default=3
        Number defining how many frames will be created for each point along the path,
        starting with the path point and interpolating to the next point.
        interpolate=1 means only showing actual path points without interpolating.
    center : float, default=(0, 0)
        Coordinates of the center of the animation.
    lim : int or float, default = 50
        Animation will be cropped to the range -lim to +lim, relative to center.
    cmap : matplotlib colormap or str, default="nipy_spectral"
        Either a colormap of choice is passed directly, or
        the name of a matplotlib colormap is passed.
    fps : float, default=12.5
        Frames per second.
    pause : int, default=2
        Number of frames the animation will freeze at the nodes, before
        showing the course of the connection again.
    size : float, default=6
        Side length of the animation in inches.
    scatter_kwargs : dict, optional
        Keyword arguments to be passed to plt.scatter(). Default entries are s=0.1, alpha=0.5
        but can be overwritten by adding them to scatter_kwargs.
    """

    Zs = graph_to_path(graph, nodes, pause)

    # convert labels to np.ndarray if necessary
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # set discrete colormap
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, len(np.unique(labels)))
    
    total_frames = ((len(Zs)-1)*interpolate + 1)
    
    # initialize figure
    fig = plt.figure(figsize=(size, size))
    plt.gca().set_aspect('equal', adjustable='box')

    # set default kwargs for scatter plot
    scatter_kwargs.setdefault("s", 0.1)
    scatter_kwargs.setdefault("alpha", 0.5)

    # initialize scatter plot
    sc = plt.scatter(*Zs[0].T, c=labels, cmap=cmap, **scatter_kwargs)

    # set axis properties
    plt.xlim([-lim + center[0], lim + center[0]])
    plt.ylim([-lim + center[1], lim + center[1]])
    plt.xticks([])
    plt.yticks([])
    plt.gca().axis("off")

    # adjust whitespace
    plt.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # define function to be called for each frame
    def update_plot(frame, interpolate):
        # interpolate between path points
        alpha = frame / interpolate
        Z = Zs[int(np.floor(alpha))] * (1 - (alpha % 1)) + Zs[int(np.ceil(alpha))] * (alpha % 1)
        
        # update all points' positions
        sc.set_offsets(Z)

        dec_places = int(np.ceil(np.log10(total_frames)))
        print(f"\rFrame {frame+1:{dec_places}}/{total_frames}", end="")

        return sc

    anim = animation.FuncAnimation(fig, update_plot, fargs=(interpolate, ), frames=total_frames, init_func=lambda: None, interval=1000/fps, repeat=False)
    return anim