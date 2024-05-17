import torch
import torch_autoneb as ta

# ==================================================
# redefine internal torch_autoneb functions


# change DataModel.forward to not need data anyore
def new_data_forward(self, dataset="train", **kwargs):
    # Apply model and use returned loss
    return self.model(**kwargs)   # new_comp_forward does not need data as input

ta.models.DataModel.forward = new_data_forward  # apply changes


# change CompareModel.forward to not use data anymore
def new_comp_forward(self, **kwargs):
    emb = self.model(**kwargs)      # this calls model.forward without arguments, just the embedding should be returned
    return self.loss(emb)

ta.models.CompareModel.forward = new_comp_forward  # apply changes


# change DataModel.analyse to not iterate over dataset, but manage dict as usual
def new_data_analyse(self):
    # Go through all data points and accumulate stats
    analysis = {}
    for ds_name, dataset in self.datasets.items():
        result = self.model.analyse() 
        for key, value in result.items():
            ds_key = f"{ds_name}_{key}"
            if ds_key not in analysis:
                analysis[ds_key] = 0
            analysis[ds_key] += value
    return analysis

ta.models.DataModel.analyse = new_data_analyse  # apply changes


# change CompareModel.analyse to just return the loss
def new_comp_analyse(self):
    # Compute loss
    emb = self.model()
    loss = self.loss(emb).item()
    return {
        "error": 0,
        "loss": loss,
    }

ta.models.CompareModel.analyse = new_comp_analyse  # apply changes

# end of redefine
# ===========================================================


class FakeModel(torch.nn.Module):
    """
    Model which does not take any input and just returns the internal embedding.
    """
    def __init__(self, num_datapoints):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(num_datapoints, 2).to(device="cuda", dtype=torch.float32).contiguous())
    def forward(self, data=None): # dummy parameter to avoid problems if data are passed
        return self.embedding # the loss has to be given as loss function to CompareModel()
    def analyse(self):
        raise NotImplementedError
    

def uniquify(path):
    """
    Find unique filenames so as to not overwrite files by accident.
    """
    import os.path
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path


def autoneb(node1, node2, loss_obj, config_path, config, initialize_count=3, graph_name="unnamed"):
    """Connect two minima on the UMAP or TSNE loss surface, optionally with a defined initial path.
        :node1 and node2: data points from x_data that shall be connected
        :loss_obj: instance of either UMAP_loss or TSNE_loss"""
    fakemodel = FakeModel(loss_obj.num_datapoints)
    lossmodel = ta.models.CompareModel(fakemodel, loss_obj)              #add loss function
    datamodel = ta.models.DataModel(lossmodel, {"train": "X"})            #add fake dataset
    model = ta.models.ModelWrapper(datamodel)                             #wrap around model

    from torch_autoneb.config import EvalConfig
    model.adapt_to_config(EvalConfig(batch_size=1))    #set batchsize for .apply() once, will be ovewritten by eval config later
    print(model.apply())   # test if model is initialized and can return a loss

    # minimize end points, create graph
    from create_graph import create_graph
    G = create_graph(node1, node2, loss_obj, config, initialize_count)

    # run AutoNEB
    lex_config = main.read_config_file(config_path, False)[3]
    ta.landscape_exploration(graph=G, model=model, lex_config=lex_config)

    # save graph
    filepath = uniquify(f"graphs/{graph_name}.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(G, f)