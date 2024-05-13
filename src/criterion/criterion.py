import torch
import torch.nn as nn
import torch.nn.functional as F


def get_criterion(config):
    if config.criterion.criterion_type == "regression":
        return nn.MSELoss()
    elif config.criterion.criterion_type == "classification":
        return nn.CrossEntropyLoss()