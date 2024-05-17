# NE-AutoNEB

```pip install``` does not work for torch_autoneb!

Instead, copy ```torch_autoneb``` folder from Felix Draxler's [PyTorch-AutoNEB](https://github.com/fdraxler/PyTorch-AutoNEB/tree/master/torch_autoneb) repository directly into the "site-packages" folder.
When using conda environments, the folder can be found inside ```envs``` at the following location:
```envs/env_name/python3.X/site-packages```.
After that, AutoNEB can be imported using ```import torch_autoneb```.

Use ```pip install -r requirements.txt``` to install all necessary packages at once.
