from torch import nn
from efficientnet_pytorch import EfficientNet


def get_net():
    net = EfficientNet.from_pretrained("efficientnet-b2")
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net
