import torch
import torch.nn.functional as F


# Override/Extend autograd calc to calculate attribution via modified chain rule
class Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias=None):
        ctx.save_for_backward(inputs, weight, bias)
        ctx.eps = 0.001
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, outputs):
        inputs, weight, bias = ctx.saved_tensors
        eps = ctx.eps

        Z = F.linear(inputs, weight, bias)

        return (
            (F.linear(outputs / (Z + eps), weight.t(), bias=None) * inputs),
            None,
            None,
        )
