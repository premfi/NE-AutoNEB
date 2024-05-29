# NE-AutoNEB

This repository contains code used for my Physics Bachelor's [Thesis](https://github.com/premfi/NE-AutoNEB/blob/main/Thesis.pdf)\
**Connecting The Dots: Low Loss Paths Between Neighbor Embeddings (2024).**

Its main functionality is based on [PyTorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB) by [Felix Draxler](https://github.com/fdraxler).



## How to install

### PyTorch-AutoNEB
```pip install``` does not work for torch_autoneb.

Instead, copy the [```torch_autoneb```](https://github.com/fdraxler/PyTorch-AutoNEB/tree/master/torch_autoneb) folder from the [PyTorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB/tree/master/torch_autoneb) repository directly into the "site-packages" folder.
When using conda environments, the folder can be found inside ```envs``` at the following location:
```envs/env_name/python3.X/site-packages```.\
After that, AutoNEB can be imported using ```import torch_autoneb```.

### PyKeOps

PyKeOps installation can be rather challenging. Using the following commands inside the activated conda environment proved to work best:\
```$ pip install git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore```\
```$ pip install git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops```

It may be necessary to add 
```export PATH=$PATH:/usr/local/cuda/bin```
to the ```.bashrc``` file in order to allow PyKeOps to find the cuda installation.

### Other

Use ```pip install -r requirements.txt``` inside the activated conda environment to install all necessary packages at once.

## Usage

The Jupyter Notebook [```connect.ipynb```](https://github.com/premfi/NE-AutoNEB/blob/main/ne_autoneb/connect.ipynb) contains an exemplary work flow to create and optimize a graph of the connection between two Neighbor Embedding minima.
Functions to plot the loss over the course of the whole optimization and to plot one or several embeddings with a colorbar can be found in [```visualize.ipynb```](https://github.com/premfi/NE-AutoNEB/blob/main/ne_autoneb/visualize.ipynb).

Loss functions for UMAP and t-SNE are included, but loss functions for other dimension reduction algorithms can be used in the same way after defining a suitable class in [```losses.py```](https://github.com/premfi/NE-AutoNEB/blob/main/ne_autoneb/losses.py).
