import torch
from torch import nn
from torch_geometric import nn as gnn
from typing import Sequence


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()


class model(nn.Module):
    def __init__(self, node_hidden_channels: Sequence):
        super().__init__()
