import os
import random
import numpy as np
import torch


def set_device(device):
    """
    Function to set the device.

    Parameters:
    -----------
    device            : str
                        Device to set.

    Returns:
    --------
    device            : torch device
                        Device to be used.
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def seed_everything(seed):
    """
    Function to set the seed for reproducibility.

    Parameters:
    -----------
    seed              : int
                        Seed to set.

    Returns:
    --------
    None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False