from typing import Tuple

import torch
from torch import nn
from torch.jit import Final


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def softplus(x):
    return nn.functional.softplus(x)


class MLP(nn.Module):
    input_dims: Final[int]  # allow jitting
    hidden_dims: Final[Tuple[int]]
    output_dim: Final[int]

    def __init__(self, input_dim: int, hidden_dims: Tuple[int], output_dim: int, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation()

        dims = [input_dim, *hidden_dims, output_dim]
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


if __name__ == "__main__":
    # can write some tests here
    model = MLP(input_dim=2, hidden_dims=(4, 4), output_dim=1)
    model = torch.jit.script(model)
    model(torch.randn(10, 2))
