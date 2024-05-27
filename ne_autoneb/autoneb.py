import torch
import torch_autoneb as ta

import redefine


class FakeModel(torch.nn.Module):
    """
    Model which does not take any input and just returns the internal embedding.
    """
    def __init__(self, num_datapoints):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(num_datapoints, 2).to(device="cuda", dtype=torch.float32).contiguous())
    def forward(self, *args, **kwargs): # dummy parameters to avoid problems if data are passed
        return self.embedding # the loss has to be given as loss function to CompareModel()
    def analyse(self):
        raise NotImplementedError
    

def read_lex_config_file(path):
    from yaml import safe_load

    with open(path, "r") as file:
        config = safe_load(file)
    return ta.config.LandscapeExplorationConfig.from_dict(config)
    

def uniquify(path):
    """
    Find unique filenames so as to not overwrite files by accident.
    """
    import os.path
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1
    return path


def autoneb(node1, node2, loss_obj, config_path, initialize=3, graph_name="unnamed_graph"):
    """Connect two minima on the UMAP or TSNE loss surface, optionally with a defined initial path.
        node1, node2 : np.ndarray or torch.Tensor
            Embeddings; data points from x_data that are to be connected.
        loss_obj : Instance of one of the classes in "losses.py"
            Object containing the precalculated high-dimensional similarities of a specific
            dataset, which is passed as parameter when initializing the object.
        config_path : str
            Path to the config file to be used for optimization during graph creation and autoneb.
        initialize : int or np.ndarray or torch.Tensor, default=3
            If int, path will be initialized by interpolating with this number of points.
            Alternatively, an arbitrary initial path can be passed, consisting of an
            arbitrary number of embeddings, excluding the node embeddings themselves.
        graph_name : str, default="unnamed_graph
            File name for the finished graph. Will be saved in folder as graphs/graph_name.pickle.
                    
                                            """
    fakemodel = FakeModel(loss_obj.num_datapoints)
    lossmodel = ta.models.CompareModel(fakemodel, loss_obj)              #add loss function
    datamodel = ta.models.DataModel(lossmodel, {"train": "X"})           #add fake dataset
    model = ta.models.ModelWrapper(datamodel)                            #wrap around model

    model.apply()   # test if model is initialized and can return a loss

    # read config file
    lex_config = read_lex_config_file(config_path)

    # minimize end points, intitialize path, create graph
    from create_graph import create_graph
    G = create_graph([node1, node2], loss_obj, lex_config, initialize)

    # run AutoNEB
    ta.landscape_exploration(graph=G, model=model, lex_config=lex_config)

    # save graph
    import pickle
    filepath = uniquify(f"graphs/{graph_name}.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(G, f)
    print("Graph successfully saved as", filepath)