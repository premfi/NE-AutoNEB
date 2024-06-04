import pickle
import os.path

import torch
import torch_autoneb as ta
from yaml import safe_load

import redefine # this import redefines internal torch_autoneb functions indirectly used in autoneb()
from create_graph import create_graph
from utils import uniquify


class FakeModel(torch.nn.Module):
    """Model that does not take any input and just returns the internal embedding."""
    def __init__(self, num_datapoints):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(num_datapoints, 2).to(device="cuda", dtype=torch.float32).contiguous())
    def forward(self, *args, **kwargs): # dummy parameters to avoid problems if data are passed
        return self.embedding # the loss has to be given as loss function to CompareModel()
    def analyse(self):
        raise NotImplementedError
    

def read_lex_config_file(path):
    with open(path, "r") as file:
        config = safe_load(file)
    return ta.config.LandscapeExplorationConfig.from_dict(config)


def autoneb(nodes, loss_inst, config_path, initialize=3, graph_name="unnamed_graph", node_idxs=None, align=True):
    """Connect two minima on the UMAP or TSNE loss surface, optionally with a predefined initial path.
        nodes : np.ndarray or torch.Tensor or list of np.ndarrays or torch.Tensors
            Embeddings; data points from x_data that are to be connected.
        loss_inst : Instance of one of the classes in "losses.py"
            Object containing the precalculated high-dimensional similarities of a specific
            dataset, which is passed as parameter during its initialization.
        config_path : str
            Path to the config file to be used for optimization during graph creation and autoneb.
        initialize : int or np.ndarray or torch.Tensor, default=3
            If int, path will be initialized by interpolating with this number of points.
            Alternatively, an arbitrary initial path can be passed, consisting of an
            arbitrary number of embeddings, excluding the node embeddings themselves.
            Path needs to be of shape (num_pivots, num_datapoints, 2).
        graph_name : str, default="unnamed_graph"
            File name for the finished graph. Will be saved in folder as graphs/graph_name.pickle.
        node_idxs : list of int, optional
            Custom indexes for the nodes of the graph. Default is ascending numbers starting at 1.
        align : bool, default=True
            If set True, embeddings will be centered and a procrustes analysis performed, that
            rotates and reflects them for maximum overlap with the first embedding. Embeddings are
            temporarily rescaled during procrustes, but are rescaled to their original size after.
            Alignment happens after pre-optimization of each embedding.
        """
    # create a fake model only containing an embedding
    fakemodel = FakeModel(loss_inst.num_datapoints)             
    lossmodel = ta.models.CompareModel(fakemodel, loss_inst)    # add loss function
    datamodel = ta.models.DataModel(lossmodel, {"train": "X"})  # add fake dataset
    model = ta.models.ModelWrapper(datamodel)                   # wrap around model

    model.apply()   # test if model is initialized and can return a loss

    # read config file
    lex_config = read_lex_config_file(config_path)

    # minimize end points, intitialize path, create graph
    G = create_graph(nodes, loss_inst, lex_config, initialize, node_idxs, align)

    # run AutoNEB
    ta.landscape_exploration(graph=G, model=model, lex_config=lex_config)

    # save graph
    filepath = uniquify(f"graphs/{graph_name}.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(G, f)
    print("Graph successfully saved as", filepath)