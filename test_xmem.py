import sys
from argparse import ArgumentParser
sys.path.append("/home/yufeiyang/Documents/XMem")

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.network import XMem


try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')



config_file = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
    'num_objects': 1,
    }

torch.autograd.set_grad_enabled(False)

model = "/home/yufeiyang/Documents/BundleSDF/BundleTrack/XMem/saves/XMem-s012.pth"
# Load our checkpoint
network = XMem(config_file, model).cuda().eval()
