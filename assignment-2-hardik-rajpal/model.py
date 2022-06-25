import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

np.random.seed(0)


class MaNet(torch.nn.Module):

    def __init__(self, imgdims, hdls, nmdigs):
        super(MaNet, self).__init__()
        self.linear1 = torch.nn.Linear(imgdims, hdls, bias=True)
        self.relu1 = torch.nn.ReLU()
        self.normalize1 = torch.nn.BatchNorm1d(hdls)
        self.linear2 = torch.nn.Linear(hdls, nmdigs, bias=True)
        self.relu2 = torch.nn.ReLU()
        self.normalize2 = torch.nn.BatchNorm1d(nmdigs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.normalize1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.normalize2(x)
        return x
