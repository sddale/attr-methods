# Gradient * input method
import torch
import numpy as np


def IntegratedGradient(inputs, target_neuron, model, device):
    with torch.device(device):
        ref_inputs = (
            torch.zeros_like(inputs).to(device).float().requires_grad_()
        )

        ref_output_neuron = model(ref_inputs)[:, target_neuron]

        grad = torch.autograd.grad(
            ref_output_neuron,
            ref_inputs,
            grad_outputs=torch.ones_like(
                ref_output_neuron, device=device
            ).float(),
            retain_graph=True,
            create_graph=True,
        )[0]

        n_steps = 10
        for alpha in np.linspace(0.1, 1.0, n_steps):
            x_tilde = ref_inputs + alpha * (inputs - ref_inputs)
            out_neuron = model(x_tilde)[:, target_neuron]
            grad += torch.autograd.grad(
                out_neuron,
                x_tilde,
                grad_outputs=torch.ones_like(out_neuron, device=device).float(),
                retain_graph=True,
                create_graph=True,
            )[0]

        return (inputs - ref_inputs) * grad / n_steps
