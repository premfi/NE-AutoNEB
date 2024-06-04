import torch
import numpy as np

import pykeops
pykeops.test_torch_bindings()
from pykeops.torch import LazyTensor

# function by Roman Remme
class ContiguousBackward(torch.autograd.Function):
    """Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction."""
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class UMAP_loss(torch.nn.Module): 
    """Class to define a UMAP loss object, precomputed on a specific dataset."""

    def __init__(self, x_data, y_data=None, UMAP_kwargs={"a":1.0, "b":1.0},
                 negative_sample_rate=5.0, push_tail=True, cuda_device=0, dtype=torch.float32):
        """Parameters
        ----------
        x_data : numpy array
            All high-dimensional data points.
        y_data : numpy array, default=None
            All labels for x_data.
        UMAP_kwargs : dict, default={"a":1.0, "b":1.0}
            Kwargs that will be passed on to umap.UMAP() .
        negative_sample_rate : float, default=5.0
            Number of negative samples per positive sample.
        push_tail : bool, default=True
            Whether tail of negative sample is pushed away from its head.
        cuda_device : int, default=0
            Index of cuda device to be used.
        dtype : torch.dtype, default=torch.float32
            Datatype to be used for torch tensors and pykeops LazyTensors. The default value should work for all functions.
        """
        super().__init__()
        import umap

        # x_data will be deleted after computing high-dimensional similarities, save length onlyTo 
        self.num_datapoints = len(x_data)
    
        # save parameters
        self.y_data = y_data
        self.push_tail = push_tail
        self.negative_sample_rate = negative_sample_rate
        self.UMAP_kwargs = UMAP_kwargs # just saving them so they can be checked later
        self.cuda_device = cuda_device
        self.dtype = dtype

        # initialize UMAP
        self.reducer = umap.UMAP(**UMAP_kwargs)
        # optimize example embedding
        self.example_embedding = self.reducer.fit_transform(x_data)

        # precompute values necessary for the loss evaluation
        self.sparse_high = self.reducer.graph_.tocoo() # create NN graph matrix in coordinate format
        torch.cuda.set_device(self.cuda_device)
        self.sparse_high_torch = torch.from_numpy(self.sparse_high.data).to(device="cuda", dtype=self.dtype)
        self.heads = torch.from_numpy(self.sparse_high.row.astype(np.int_)).to(device="cuda")
        self.tails = torch.from_numpy(self.sparse_high.col.astype(np.int_)).to(device="cuda")
        
        # get decreased repulsive weights
        self.push_weights = self.get_UMAP_push_weight_keops(self.sparse_high)

        print("UMAP_loss initialized successfully")

    def low_dim_sim_keops_dist(self, x, squared=True):
        """Smooth function from distances to low-dimensional simiarlity. Compatible with keops"""
        if not squared:
            return 1.0 / (1.0 + x ** 2.0)
        return 1.0 / (1.0 + x) 
    
    def compute_low_dim_psim_keops_embd(self, embedding):
        """Computes low-dimensional pairwise similarites from embeddings via keops."""
        lazy_embd_i = LazyTensor(embedding[:, None, :].to(dtype=self.dtype))
        lazy_embd_j = LazyTensor(embedding[None].to(dtype=self.dtype))

        sq_dists = ((lazy_embd_i-lazy_embd_j) ** 2).sum(-1)
        return 1.0 / (1.0 + sq_dists) # squared = True
    
    def get_UMAP_push_weight_keops(self, high_sim):
        """Computes the effective, decreased repulsive weights and the degrees of each node, keops implementation"""
        n_points = LazyTensor(torch.tensor(high_sim.shape[0], device="cuda", dtype=self.dtype).contiguous())

        degrees = np.array(high_sim.sum(-1)).ravel()
        degrees_t = torch.tensor(degrees, device="cuda", dtype=self.dtype).contiguous()
        
        degrees_i = LazyTensor(degrees_t[:, None, None])
        degrees_j = LazyTensor(degrees_t[None, :, None])

        if self.push_tail:
            return self.negative_sample_rate * (degrees_i + degrees_j)/(2*n_points)
        return self.negative_sample_rate * degrees_i * LazyTensor(torch.ones((1,len(degrees_t), 1), device="cuda").contiguous())/n_points
    
    # adapted from: https://github.com/sdamrich/vis_utils/blob/main/vis_utils/utils.py#L723
    def forward(self, embedding, pointwise=False, eps=0.0001):
        """UMAP's true loss function, keops implementation"""

        sq_dist_pos_edges = ((embedding[self.heads]-embedding[self.tails])**2).sum(-1)               # performance ~1.43ms
    
        low_sim_pos_edges = self.low_dim_sim_keops_dist(sq_dist_pos_edges, squared=True)  # performance ~640µs

        low_sim = self.compute_low_dim_psim_keops_embd(embedding)                         # performance ~61µs

        loss_a_unsummed = (self.sparse_high_torch * torch.log(low_sim_pos_edges))
        loss_a = loss_a_unsummed.sum()
        
        # get decreased repulsive weights
        push_weights = self.get_UMAP_push_weight_keops(self.sparse_high)#[0])  # performance ~150µs

        inv_low_sim = 1 - (low_sim - eps).relu() # pykeops compatible version of min(1-low_sim+eps, 1)

        loss_r_unsummed = ContiguousBackward().apply((push_weights * inv_low_sim.log()).sum(1)) # was  ).sum(1))
        loss_r = loss_r_unsummed.sum()             # performance ~10.3ms

        if pointwise:
            return -loss_a + (-loss_r), loss_a_unsummed, loss_r_unsummed
        else:
            return -loss_a + (-loss_r)


class TSNE_loss(torch.nn.Module):
    """Class to define a KL Divergence loss object, precomputed on a specific dataset."""

    def __init__(self, x_data, y_data=None, scale=3.5, TSNE_kwargs={}, cuda_device=0, dtype=torch.float32):
        """Parameters
        ----------
        x_data : numpy array
            All high-dimensional data points.
        y_data : numpy array, default=None
            All labels for x_data.
        scale : float, default=3.5
            Embeddings will be multiplied with this value before calculating its loss. Learning rates
            used with default openTSNE need to be divided by scale before used with TSNE_loss to be comparable,
            as the gradients of the loss will be scaled together with the embeddings.
        TSNE_kwargs : dict, default={"a":1.0, "b":1.0}
            Kwargs that will be passed on to openTSNE.affinity.PerplexityBasedNN().
        cuda_device : int, default=0
            Index of cuda device to be used.
        dtype : torch.dtype, default=torch.float32
            Datatype to be used for torch tensors and pykeops LazyTensors. The default
            value should work for all functions.
        """
        super().__init__()
        import openTSNE
        import scipy

        # x_data will be deleted after computing high-dimensional similarities, save length only
        self.num_datapoints = len(x_data)

        # save parameters
        self.y_data = y_data
        self.scale = scale
        self.TSNE_kwargs = TSNE_kwargs
        self.cuda_device = cuda_device
        self.dtype = dtype

        # compute high dimensional similarities
        high_sim_all = openTSNE.affinity.PerplexityBasedNN(x_data, **TSNE_kwargs)
        sparse_high = scipy.sparse.coo_matrix(high_sim_all.P)

        # precompute the necessary values for the loss evaluation
        torch.cuda.set_device(self.cuda_device)
        self.sparse_high_data_torch = torch.from_numpy(sparse_high.data).to(device="cuda", dtype=self.dtype)
        self.high_sim_pos_edges_norm = torch.from_numpy(sparse_high.data / sparse_high.data.sum()).to(device="cuda", dtype=self.dtype, memory_format=torch.contiguous_format)
        self.heads = torch.from_numpy(sparse_high.row.astype(np.int_)).to(device="cuda")
        self.tails = torch.from_numpy(sparse_high.col.astype(np.int_)).to(device="cuda")
        
        print("TSNE_loss initialized successfully")

    def low_dim_sim_keops_dist(self, x, squared=True):
        """Smooth function from distances to low-dimensional simiarlity. Compatible with keops"""
        if not squared:
            return 1.0 / (1.0 + x ** 2.0)
        return 1.0 / (1.0 + x) 
    
    def compute_low_dim_psim_keops_embd(self, embedding):
        """Computes low-dimensional pairwise similarites from embeddings via keops."""
        lazy_embd_i = LazyTensor(embedding[:, None, :].to(dtype=self.dtype))
        lazy_embd_j = LazyTensor(embedding[None].to(dtype=self.dtype))

        sq_dists = ((lazy_embd_i-lazy_embd_j) ** 2).sum(-1)
        return 1.0 / (1.0 + sq_dists) # squared = True
    
    def keops_identity(self, n):
        x = torch.arange(n, dtype=self.dtype, device="cuda")

        x_i = LazyTensor(x[:, None], axis=0)
        x_j = LazyTensor(x[:, None], axis=1)

        id_mat  = (0.5-(x_i-x_j).abs()).step()
        return id_mat
    
    def compute_normalization(self, x, no_diag=True):
        """Pykeops implementation for computing"""
        sims = self.compute_low_dim_psim_keops_embd(x)

        if no_diag:
            sims = sims * ( 1.0 - self.keops_identity(len(x)) )

        total_sim = ContiguousBackward().apply(sims.sum(1)).sum(0)  # Performance ~10.7ms
        return total_sim

    # adapted from: https://github.com/sdamrich/vis_utils/blob/main/vis_utils/utils.py#L631
    def forward(self, embedding,  # torch.tensor
                    scale=3.5):                # openTSNE embeddings are enlargened by this loss, scaling them up beforehands prevents that
        """Computes the KL divergence between the high-dimensional p and low-dimensional
        similarities q. The latter are inferred from the embedding.
        """
        embedding = embedding * scale

        sq_dist_pos_edges = ((embedding[self.heads]-embedding[self.tails])**2).sum(-1)  # Performance ~1.44ms

        low_sim_pos_edges = self.low_dim_sim_keops_dist(sq_dist_pos_edges, squared=True)  # Performance <1ms

        total_low_sim = self.compute_normalization(embedding, no_diag=True)             # Performance ~10.8ms
        low_sim_pos_edges_norm = low_sim_pos_edges / total_low_sim

        neg_entropy = (self.high_sim_pos_edges_norm * torch.log(self.high_sim_pos_edges_norm)).sum()   # Performance ~378µs
        cross_entropy = - (self.high_sim_pos_edges_norm * torch.log(low_sim_pos_edges_norm)).sum()     # Performance ~379µs
        return (cross_entropy + neg_entropy)
