import os.path

import torch
import numpy as np
from scipy.spatial import procrustes


def uniquify(path):
    """Find unique filenames so as to not overwrite files by accident."""
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1
    return path


def to_array(embs, dtype=np.float64):
    """Converts iterable of embeddings to 3D numpy array of given dtype.

    Parameters
    ----------
    embs : np.ndarray, torch.Tensor, list of np.array or torch.Tensor
        The iterable of embeddings to be converted. If only one embedding is passed (2D),
        an extra dimension will be added.
    dtype : str or np.dtype, default=np.float64
    """
    if isinstance(embs, (list, tuple)):
        if isinstance(embs[0], np.ndarray): # list or tuple of np.ndarrays
            embs = np.array(embs)
        elif isinstance(embs[0], torch.Tensor): # list or tuple of torch.Tensors
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
    
    return embs.astype(dtype)


def align_embs(embs, standard=0):
    """Centers and rotation aligns all embeddings using procrustes analysis."""
    embs = to_array(embs)

    # center embeddings
    embs -= np.average(embs, axis=1, keepdims=True)

    rotated = np.zeros_like(embs[1:])
    for i, emb in enumerate(np.concatenate((embs[:standard], embs[standard + 1:]))):
        # rotate and reflect emb for maximum overlap with standard embedding
        emb_rot = procrustes(embs[standard], emb)[1]
        # rescale rotated embedding to original size
        rotated[i] = procrustes(emb_rot, emb_rot)[0] * (emb / procrustes(emb, emb)[0])

    return np.concatenate((rotated[:standard], embs[standard][None], rotated[standard:]))