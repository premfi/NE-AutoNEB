import itertools

import torch
import numpy as np
import networkx as nx

from utils import align_embs


def optimize(emb, loss_inst, config):
    """Optimize one single embedding in the exact same way as specified in config."""
    if type(emb) == np.ndarray:
        emb = torch.from_numpy(emb)

    # make tensors live on same cuda device as the loss function evaluating them
    torch.cuda.set_device(loss_inst.cuda_device)

    # initialize embedding to be optimized
    path_point = emb.to(device="cuda", dtype=loss_inst.dtype)
    path_point.requires_grad_(True)

    # initialize optimizer
    optimizer = torch.optim.SGD([path_point], lr=0.001, momentum=0.9)

    # iterate through config cycles
    for config_cycle in config.auto_neb_config.neb_configs:
        # adapt to config
        nsteps = config_cycle.optim_config.nsteps
        optimizer.param_groups[0]["lr"] = config_cycle.optim_config.algorithm_args["lr"]
        optimizer.param_groups[0]["momentum"] = config_cycle.optim_config.algorithm_args["momentum"]
    
        # optimize for given number of steps during each cycle
        for i in range(nsteps):
            optimizer.zero_grad()

            # apply the loss function defined by loss_inst
            loss = loss_inst(path_point)
            loss.backward()

            # only clip gradients if clip_grad is not set to 0
            if config_cycle.optim_config.clip_grad:
                torch.nn.utils.clip_grad_value_(path_point, config_cycle.optim_config.clip_grad)

            optimizer.step()
    
    # graph should be saved on cpu only
    return path_point.detach().to(device="cpu"), loss.item()


def create_graph(nodes, loss_inst, config, initialize, node_idxs=None, align=True):
    """Create graph, initialize edges with linear interpolation."""
    # use ascending integers as node indeces if not specified otherwise
    if node_idxs is None:
        node_idxs = np.arange(len(nodes)) + 1

    # optimize nodes
    nodes_optimized = []
    losses_optimized = []
    for node in nodes:
        node_optimized, loss_optimized = optimize(node, loss_inst, config) # returns a tuple: (node, loss)
        nodes_optimized = nodes_optimized + [node_optimized] # list of torch.Tensors
        losses_optimized = losses_optimized + [loss_optimized]

    # convert nodes to numpy arrays if they are not already
    nodes = [node.numpy() if (type(node) == torch.Tensor) else node for node in nodes]

    # align nodes
    if align:
        nodes = align_embs(nodes)
        nodes_optimized = align_embs(nodes_optimized)
        # alignment returns np.ndarray, but torch.Tensor is needed later
        nodes_optimized = torch.from_numpy(nodes_optimized).to(dtype=loss_inst.dtype)

    # nodes_optimized does now contain torch.Tensors, regardless of if align was executed

    # create graph with nodes containing aligned and optimized embeddings
    G = nx.MultiGraph()
    for i, (node_optimized, loss_optimized) in enumerate(zip(nodes_optimized, losses_optimized)):
        G.add_node(node_idxs[i], coords=node_optimized, train_loss=loss_optimized, train_error=0)

    # check if number of interpolations or an initial path was passed
    try:
        initialize = int(initialize)
    except TypeError:
        # if multiple nodes were passed, initialize linearly even if an initial path was given
        if len(nodes) > 2:
            initialize = len(initialize)
            print(f"Custom initialization is not supported for multiple nodes. Using linear initialization with {initialize} pivots instead.")
        # convert initialize to torch.Tensor if it was given as np.ndarray
        if isinstance(initialize, np.ndarray):
            initialize = torch.from_numpy(initialize).to(dtype=loss_inst.dtype)

    for i, j in itertools.combinations(range(len(nodes)), 2):
        # define initial path
        if isinstance(initialize, int):
            # initialize by interpolating linearly between non-optimized node embeddings with initialize steps
            # expects nodes to be np.ndarrays, nodes_optimized to contain torch.Tensor
            fake_path_coords = np.array([(1 - alpha) * nodes[i] + alpha * nodes[j] for alpha in np.linspace(0, 1, initialize+2)[1:-1]])
            fake_path_coords = torch.cat((nodes_optimized[i][None], torch.from_numpy(fake_path_coords).to(dtype=loss_inst.dtype), nodes_optimized[j][None]))

        else:
            # initialize path with the points given as initialize
            # expects initialize to be torch.Tensor, nodes_optimized to contain torch.Tensor
            fake_path_coords = torch.cat((nodes_optimized[i][None], initialize, nodes_optimized[j][None])) # for predefined path initialization
            
        # reshape each point to 1D tensor
        fake_path_coords = fake_path_coords.reshape((len(fake_path_coords), 2 * loss_inst.num_datapoints)).cpu()

        # create edge to make AutoNEB believe that an optimization cycle already happened, resulting in this edge
        fake_edge_dict = {"path_coords": fake_path_coords,
                        'target_distances': (fake_path_coords[:-1] - fake_path_coords[1:]).norm(2, 1),
                        'saddle_train_error': 0.0,
                        'saddle_train_loss': 0.0,}
        
        # add edge with cycle number 0. AutoNEB would normally start the cycle numbers with 1
        G.add_edge(node_idxs[i], node_idxs[j], 0, **fake_edge_dict)

    return G