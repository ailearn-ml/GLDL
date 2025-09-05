import numpy as np
import warnings
import random
import torch

warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms