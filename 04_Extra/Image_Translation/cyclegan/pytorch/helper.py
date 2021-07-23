import torch
from torch import nn

def set_requires_grad(models:list, requires_grad:bool) -> None:
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad

