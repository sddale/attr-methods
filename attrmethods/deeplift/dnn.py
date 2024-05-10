# simple dnn
import torch
from torch import nn
from .linear import Linear
from .sequential import Sequential

# activation = nn.ReLU
activation = nn.Tanh
# activation = nn.Softplus
# activation = nn.Sigmoid


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        bias = False
        self.f = Sequential(
            nn.Flatten(),
            Linear(28 * 28, 512, bias=bias),
            activation(),
            Linear(512, 512, bias=bias),
            activation(),
            Linear(512, 10, bias=bias),
        )

    def forward(self, x, explain=False):
        return self.f(x, explain=explain)
