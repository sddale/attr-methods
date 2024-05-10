import torch
from .linear import Linear


# Sequential NN container modified to calculate attribute when explain=True
class Sequential(torch.nn.Sequential):
    def forward(self, inputs, explain=False):
        if not explain:  # just pass to standard torch sequential
            return super(Sequential, self).forward(inputs)

        for module in self:
            if isinstance(module, Linear):
                inputs, _ = module.forward(inputs, explain=True)
            else:  # No attr overload for these layers, so just pass along inputs
                inputs = module(inputs)

        return inputs
