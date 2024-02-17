## useful tools for pets segmentation
import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
working_dir = "./pets/working/"

# save model status and checkpoint
def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))

# get the current device: cpu or gpu
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model from saved checkpoint
# If map_location is a torch.device object or a string containing a device tag, it indicates the location where all tensors should be loaded.
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )

# send tensor and model to the device
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

# get the total number of parameters of the model
def get_model_parameters(m): # numel: total number of elements
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

# print the total number of parameters
def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")

# close all the figures that are opened in the server
def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()