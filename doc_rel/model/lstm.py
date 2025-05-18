import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        hid_channels: int = 64,
        num_classes: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.name = "lstm"

        self.gru_layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=hid_channels,
            num_layers=2,
            bias=True,
            dropout=0.3,
            batch_first=True,
        )

        self.out = nn.Linear(hid_channels, num_classes)

        self.gru_layer.flatten_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        r_out, (h, c) = self.gru_layer(x, None)
        x = self.out(h[-1, :, :])
        x = x.reshape((x.shape[0],))
        return x
