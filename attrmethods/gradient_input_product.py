# Gradient * input method
import torch
import numpy as np


def GradientInputProduct(inputs, target_neuron, model, device):
    with torch.device(device):
        target_neuron = model(inputs)[:, target_neuron]

        grad = torch.autograd.grad(
            target_neuron,
            inputs,
            grad_outputs=torch.ones_like(target_neuron, device=device).float(),
            retain_graph=True,
            create_graph=True,
        )[0]

        return inputs * grad
