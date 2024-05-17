import networkx as nx
import torch

def optimize(emb, loss_obj, config):
    """
    Optimize one single embedding.
    """
    nsteps = config["nsteps"]     # int
    initial_lr = config["lr"]     # float
    lr_headstart = config["lr_headstart"] # int
    lr_decay = config["lr_decay"] # bool
    lr_interval = config["lr_interval"]   # int
    momentum = config["momentum"] # 1 > float >= 0

    # calculate for later
    lr_snap = (nsteps - lr_headstart) / lr_interval
    clip_grad = loss_obj.clip_grad

    path_point = torch.from_numpy(emb).to(device="cuda", dtype=torch.float32)
    path_point.requires_grad_(True)

    optimizer = torch.optim.SGD([path_point], lr=initial_lr, momentum=momentum) # with optim
    for i in tqdm.tqdm(range(nsteps), f"Pre-optimizing embedding"):
        optimizer.zero_grad() # with optim

        # evaluate loss and backpropagate through it
        loss = loss_obj(path_point)
        loss.backward()

        if lr_decay:
            if i < lr_headstart:
                lr = initial_lr
            else:
                lr = (np.ceil(((nsteps-i) / (nsteps-lr_headstart)) * lr_snap) / lr_snap) * initial_lr  #lr decay, fitting UMAP5+4
        else:
            lr = initial_lr

        if clip_grad:
            torch.nn.utils.clip_grad_value_(path_point, clip_grad)

        # take one step with current lr
        optimizer.param_groups[0]['lr'] = lr
        optimizer.step()

    return path_point.detach().to(device="cpu")


def create_graph(node1, node2, loss_obj, config, initialize_count):
    """
    Create graph, initialize edges with linear interpolation.
    """
    # create graph with nodes containing rotated and optimized embeddings
    G = nx.MultiGraph()
    G.add_node(1, coords=node1, train_loss=0, train_error=0) # the node attributes do not matter in the optimization, thus dummy values
    G.add_node(2, coords=node1, train_loss=0, train_error=0)

    # initialize path with linear interpolation between non-optimized original embeddings
    fake_path_coords = np.array([(1 - alpha) * node1 + alpha * node2 for alpha in np.linspace(0, 1, initialize_count+2)[1:-1]])
    fake_path_coords = torch.cat((G.nodes[i0]["coords"][None], torch.from_numpy(fake_path_coords).to(dtype=torch.float32), G.nodes[i1]["coords"][None])) #device="cuda", 
    fake_path_coords = fake_path_coords.reshape((initialize_count + 2, 2*num_datapoints)).cpu()
    fake_edge_dict = {"path_coords": fake_path_coords,
                    'target_distances': (fake_path_coords[:-1] - fake_path_coords[1:]).norm(2, 1),
                    'saddle_train_error': 0.0,
                    'saddle_train_loss': 0.0,}
    
    G.add_edge(i0, i1, 0, **fake_edge_dict)
    return G