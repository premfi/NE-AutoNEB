# NE-AutoNEB

## How to install
```pip install``` does not work for torch_autoneb!

Instead, copy ```torch_autoneb``` folder from Felix Draxler's [PyTorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB/tree/master/torch_autoneb) repository directly into the "site-packages" folder.
When using conda environments, the folder can be found inside ```envs``` at the following location:
```envs/env_name/python3.X/site-packages```.
After that, AutoNEB can be imported using ```import torch_autoneb```.

Use ```pip install -r requirements.txt``` to install all necessary packages at once.

## Usage

The Jupyter Notebook [```connect.ipynb```](https://github.com/premfi/NE-AutoNEB/blob/main/ne_autoneb/connect.ipynb) contains an exemplary work flow to create and optimize a graph of the connection between two Neighbor Embedding minima.
Basic functions to plot the loss over the course of the whole optimization and to plot a single embedding with colorbar can be found in [```visualize.ipynb```](https://github.com/premfi/NE-AutoNEB/blob/main/ne_autoneb/visualize.ipynb).
