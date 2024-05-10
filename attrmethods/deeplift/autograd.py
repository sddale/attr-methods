import torch
import torch.nn.functional as F

nonlinearity = torch.nn.Sigmoid


# Override/Extend autograd calc to calculate attribution via modified chain rule
class Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, ref_input, weight, bias=None):
        ctx.save_for_backward(inputs, ref_input, weight, bias)
        ctx.eps = 0.001
        return F.linear(inputs, weight, bias), F.linear(ref_input, weight, bias)

    @staticmethod
    def backward(ctx, outputs, ref_outputs):
        inputs, ref_input, weight, bias = ctx.saved_tensors
        eps = ctx.eps

        Z = F.linear(inputs, weight, bias)
        Z_ref = F.linear(ref_input, weight, bias)

        return (
            F.linear(
                (outputs - ref_outputs) / (Z - Z_ref + eps),
                weight.t(),
                bias=None,
            )
            * (inputs - ref_input),
            None,
            None,
            None,
        )
