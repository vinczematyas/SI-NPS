import sys
import json
import numpy as np
import torch
import random


def custom_argparser():
    argv_dict = {}
    for arg in sys.argv[1:]:
        key, sep, value = arg.partition("=")
        if value in ["True", "False"]:
            value = value == "True"
        elif value.isdigit():
            value = int(value)
        elif "[" in value or "]" in value:
            value = json.loads(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        argv_dict[key[2:]] = value
    return argv_dict


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
