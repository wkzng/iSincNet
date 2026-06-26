import random
import numpy as np
import os
import torch


def set_random_seed(seed:int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_gradients(model:torch.nn.Module, clip:float):
    """Rescale norm of computed gradients"""
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def custom_load_state_dict(model:torch.nn.Module, pretrained_dict:dict) -> torch.nn.Module:
    """ load only matching keys """
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    commons = set(model_dict.keys()).intersection(set(pretrained_dict.keys()))
    print("==============COMMON=======================")
    for k in sorted(commons):
        print(k)

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model
