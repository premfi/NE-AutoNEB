import torch
import torch_autoneb as ta


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


# change neb to optionally apply gradient clipping
def neb_edited(previous_cycle_data, model, neb_config):
    # Initialise chain by inserting pivots
    start_path, target_distances = neb_config.insert_method(previous_cycle_data, **neb_config.insert_args)

    # Model
    neb_mod = ta.neb_model.NEB(model, start_path, target_distances)
    neb_mod.adapt_to_config(neb_config)

    # Load optimiser
    optim_config = neb_config.optim_config
    # HACK: Optimisers only like parameters registered to autograd -> proper solution would keep several model instances as path and nudge their gradients after backward.
    neb_mod.path_coords.requires_grad_(True)
    optimiser = optim_config.algorithm_type(neb_mod.parameters(), **optim_config.algorithm_args)  # type : optim.Optimizer
    # HACK END: We don't want autograd to mingle with our computations
    neb_mod.path_coords.requires_grad_(False)
    if "weight_decay" in optimiser.defaults:
        assert optimiser.defaults["weight_decay"] == 0, "NEB is not compatible with weight decay on the optimiser. Set weight decay on NEB instead."

    # Scheduler
    if optim_config.scheduler_type is not None:
        scheduler = optim_config.scheduler_type(optimiser, **optim_config.scheduler_args)
    else:
        scheduler = None

    # Optimise
    for _ in ta.helpers.pbar(range(optim_config.nsteps), "NEB"):
        neb_mod.apply(gradient=True)
        # Clip gradients as specified in config by optional argument "clip_grad"
        if optim_config.clip_grad:                                                       # ADDED
            torch.nn.utils.clip_grad_value_(neb_mod.path_coords, optim_config.clip_grad) # ADDED
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
    result = {
        "path_coords": neb_mod.path_coords.clone().to("cpu"),
        "target_distances": target_distances.to("cpu")
    }

    # Analyse
    analysis = neb_mod.analyse(neb_config.subsample_pivot_count)
    saddle_analysis = {key: value for key, value in analysis.items() if "saddle_" in key}
    ta.logger.debug(f"Found saddle: {saddle_analysis}.")
    result.update(analysis)
    return result

ta.neb = neb_edited  # apply changes


# change OptimConfig to include parameter clip_grad
class OptimConfig_edited(ta.config.BaseConfig):
    def __init__(self, nsteps, clip_grad, algorithm_type, algorithm_args, scheduler_type, scheduler_args, eval_config):
        self.nsteps = nsteps
        self.clip_grad = clip_grad    # ADDED
        self.algorithm_type = algorithm_type
        self.algorithm_args = algorithm_args
        self.scheduler_type = scheduler_type
        self.scheduler_args = scheduler_args
        self.eval_config = eval_config

    @staticmethod
    def from_dict(config_dict: dict):
        from copy import deepcopy
        config_dict = deepcopy(config_dict)
        config_dict["algorithm_type"], config_dict["algorithm_args"] = ta.config.replace_instanciation(config_dict["algorithm"], torch.optim)
        del config_dict["algorithm"]
        if "scheduler" in config_dict:
            config_dict["scheduler_type"], config_dict["scheduler_args"] = ta.config.replace_instanciation(config_dict["scheduler"], torch.optim.lr_scheduler)
            del config_dict["scheduler"]
        else:
            config_dict["scheduler_type"], config_dict["scheduler_args"] = None, None
        if "eval" in config_dict:
            config_dict["eval_config"] = ta.config.EvalConfig.from_dict(config_dict["eval"])
            del config_dict["eval"]
        else:
            config_dict["eval_config"] = None        
        if "clip_grad" not in config_dict:      # ADDED
            config_dict["clip_grad"] = 0        # ADDED
        return OptimConfig_edited(**config_dict)
    
ta.config.OptimConfig = OptimConfig_edited  # apply changes