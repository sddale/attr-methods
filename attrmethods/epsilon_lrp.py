# Gradient * input method
import torch
import numpy as np
from .lrp import DNN


def EpsilonLRP(inputs, target_neuron, model, device):

    # Transfer learning to modfied DNN with attr layers
    dnn = DNN().to(device)
    dnn.load_state_dict(model.state_dict())

    with torch.no_grad():
        for p1, p2 in zip(
            model.parameters(),
            dnn.parameters(),
        ):
            p2.data.copy_(p1.data)

    # Backwards modified autograd
    inputs.grad = None
    inputs.retain_grad()
    outputs = dnn(inputs, explain=True)[:, target_neuron]
    outputs.backward()

    return inputs.grad
