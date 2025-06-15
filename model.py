import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pytorch_model_summary import summary

from config import Config
from utils import get_device


def create_swin_transformer():
    device = get_device()    
    net = timm.create_model(Config.MODEL_NAME, pretrained=True)
    net.head.fc = nn.Linear(
        in_features=Config.MODEL_INPUT_FEATURES, 
        out_features=Config.MODEL_OUTPUT_FEATURES, 
        bias=True
    )
    net = net.to(device)
    return net