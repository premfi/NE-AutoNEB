import torch
import numpy as np
import networkx as nx

def optimize(emb, loss_obj, config):
    """
    Optimize one single embedding in the exact same way as specified in config.
    """
    if type(emb) == np.ndarray:
        emb = torch.from_numpy(emb)

    # make tensors live on same cuda device as the loss function evaluating them
    torch.cuda.set_device(loss_obj.cuda_device)

    # initialize embedding to be optimized
    path_point = emb.to(device="cuda", dtype=loss_obj.dtype)
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

            # apply the loss function defined by loss_obj
            loss = loss_obj(path_point)
            loss.backward()

            # only clip gradients if clip_grad is not set to 0
            if config_cycle.optim_config.clip_grad:
                torch.nn.utils.clip_grad_value_(path_point, config_cycle.optim_config.clip_grad)

            optimizer.step()
    
    # graph should be saved on cpu only
    return path_point.detach().to(device="cpu"), loss.item()



def create_graph(nodes, loss_obj, config, initialize):
    """
    Create graph, initialize edges with linear interpolation.
    """
    node1, node2 = nodes
    node_names = [1, 2]

    # optimize nodes
    node1_optimized, node1_loss = optimize(node1, loss_obj, config)
    node2_optimized, node2_loss = optimize(node2, loss_obj, config)

    # convert nodes to numpy arrays if they are not already
    nodes = [torch.from_numpy(node)if (type(node) == np.ndarray) else node for node in nodes]

    # create graph with nodes containing rotated and optimized embeddings
    G = nx.MultiGraph()
    G.add_node(1, coords=node1_optimized, train_loss=node1_loss, train_error=0) # the node attributes do not matter for optimization
    G.add_node(2, coords=node2_optimized, train_loss=node2_loss, train_error=0)

    # check if number of interpolations or an initial path was passed
    try:
        initialize = int(initialize)
    except TypeError:
        # convert initialize to torch.Tensor if it was given as np.ndarray
        if isinstance(initialize, np.ndarray):
            initialize = torch.from_numpy(initialize).to(dtype=loss_obj.dtype)

    # define initial path
    if isinstance(initialize, int):
        # initialize by interpolating linearly between non-optimized node embeddings with initialize steps
        # expects node1 and node2 to be np.ndarrays
        fake_path_coords = np.array([(1 - alpha) * node1 + alpha * node2 for alpha in np.linspace(0, 1, initialize+2)[1:-1]])
        fake_path_coords = torch.cat((node1_optimized[None], torch.from_numpy(fake_path_coords).to(dtype=loss_obj.dtype), node2_optimized[None]))

    else:
        # initialize path with the points given as initialize
        # expects initialize to be torch.Tensor
        fake_path_coords = torch.cat((node1_optimized[None], initialize, node2_optimized[None])) # for predefined path initialization
        
    # reshape each point to 1D tensor
    fake_path_coords = fake_path_coords.reshape((len(fake_path_coords), 2 * loss_obj.num_datapoints)).cpu()

    # create edge to make AutoNEB believe that an optimization cycle already happened, resulting in this edge
    fake_edge_dict = {"path_coords": fake_path_coords,
                    'target_distances': (fake_path_coords[:-1] - fake_path_coords[1:]).norm(2, 1),
                    'saddle_train_error': 0.0,
                    'saddle_train_loss': 0.0,}
    
    # add edge with cycle number 0. AutoNEB would normally start the cycle numbers with 1
    G.add_edge(node_names[0], node_names[1], 0, **fake_edge_dict)

    return G