import torch
from .autograd import Autograd


# Linear layer modified to apply autograd override
class Linear(torch.nn.Linear):
    def forward(self, inputs, explain=False):
        if not explain:
            return super(Linear, self).forward(inputs)
        return Autograd.apply(inputs, self.weight, self.bias)
