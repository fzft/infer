import torch
import torch.nn as nn
import pnnx


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

model = Model()


if __name__ == "__main__":
    model = Model()
    x = torch.randn(10)
    opt_model = pnnx.export(model, "model.pt", x)
