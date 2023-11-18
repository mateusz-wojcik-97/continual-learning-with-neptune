import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_layer_size: int = 784, hidden_layer_size: int = 256, output_layer_size: int = 10, drop_rate: float = 0.5) -> None:
        super().__init__()
        self._input_layer_size = input_layer_size
        self._layers = nn.Sequential(
            nn.Linear(
                in_features=input_layer_size,
                out_features=hidden_layer_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(
                in_features=hidden_layer_size,
                out_features=output_layer_size,
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        x = x.view(x.size(0), self._input_layer_size)
        return self._layers(x)
