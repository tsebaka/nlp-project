import torch
from transformers import AutoConfig
from .base_model import BaseModel
from transformers import AutoModelForSequenceClassification


def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
        

def get_model(config, train=True):
    model = AutoModelForSequenceClassification.from_pretrained(config.model.path, num_labels=1)
    return model