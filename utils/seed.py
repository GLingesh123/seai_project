import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """
    Make runs reproducible across:
    - python random
    - numpy
    - torch CPU
    - torch CUDA
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic torch behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to {seed}")
