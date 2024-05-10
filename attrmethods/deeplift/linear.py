import torch
from .autograd import Autograd


# Linear layer modified to apply autograd override
class Linear(torch.nn.Linear):
    def forward(self, inputs, ref_input=None, explain=False):
        if not explain:
            return super(Linear, self).forward(inputs)

        if ref_input is None:
            ref_input = torch.zeros_like(inputs).float().requires_grad_()
        return Autograd.apply(inputs, ref_input, self.weight, self.bias)
