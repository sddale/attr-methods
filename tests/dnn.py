# simple dnn
import torch
from torch import nn

# activation = nn.ReLU
activation = nn.Tanh
# activation = nn.Softplus
# activation = nn.Sigmoid


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        bias = False
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512, bias=bias),
            activation(),
            nn.Linear(512, 512, bias=bias),
            activation(),
            nn.Linear(512, 10, bias=bias),
        )

    def forward(self, x, explain=False):
        return self.f(x)

    def train_model(self, epochs, loaders, device):
        with torch.device(device):
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                self.parameters(), lr=3e-3, momentum=0.9
            )

            self.train()

            n_batch = len(loaders["train"])
            for epoch in range(epochs):
                for i, (images, labels) in enumerate(loaders["train"], start=1):
                    loss = loss_func(
                        self(images.to(device).float().requires_grad_()),
                        labels.to(device).float().requires_grad_(),
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{n_batch}],"
                            + f"Loss: {loss.item():.5f}".format()
                        )
