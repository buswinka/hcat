import torch.nn as nn
from torch import Tensor
from typing import *


class DeepABRModel(nn.Module):
    def __init__(self, n_waveforms: int = 20, channels: List[int] = [15, 25, 35, 45]):
        super(DeepABRModel, self).__init__()

        # input (1, 1, 20, 422) CLIP AT 252
        self.layer_0 = nn.Conv2d(
            1, channels[0], kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)
        )
        self.layer_1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)
        )
        self.layer_2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)
        )
        self.layer_3 = nn.Conv2d(
            channels[2], channels[3], kernel_size=(3, 9), stride=(1, 1), padding=(1, 4)
        )

        # 256 -> 128
        self.pool0 = nn.Conv2d(
            channels[0], channels[0], kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )
        # 128 -> 64
        self.pool1 = nn.Conv2d(
            channels[1], channels[1], kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )
        # 64 -> 32
        self.pool2 = nn.Conv2d(
            channels[2], channels[2], kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )
        # 32 -> 16
        self.pool3 = nn.Conv2d(
            channels[3], channels[3], kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)
        )

        self.linear0 = nn.Linear(channels[3] * n_waveforms * 16, 10 * n_waveforms * 16)
        self.linear1 = nn.Linear(10 * n_waveforms * 16, 5 * n_waveforms * 16)
        self.linear2 = nn.Linear(5 * n_waveforms * 16, n_waveforms * 16)
        self.linear3 = nn.Linear(16 * n_waveforms, 22)

        self.norm0 = nn.LayerNorm([channels[0], n_waveforms, 256])
        self.norm1 = nn.LayerNorm([channels[1], n_waveforms, 256 // 2])
        self.norm2 = nn.LayerNorm([channels[2], n_waveforms, 256 // 4])
        self.norm3 = nn.LayerNorm([channels[3], n_waveforms, 256 // 8])

        self.activation = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

        self.device = 'cpu'

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.pool0(self.norm0(self.activation(self.layer_0(x)))))
        x = self.activation(self.pool1(self.norm1(self.activation(self.layer_1(x)))))
        x = self.activation(self.pool2(self.norm2(self.activation(self.layer_2(x)))))
        x = self.activation(self.pool3(self.norm3(self.activation(self.layer_3(x)))))
        x = x.view(x.shape[0], -1)
        x = self.activation(self.linear0(x))
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return self.softmax(x)

    def to(self, device):
        self.device = device
        super(DeepABRModel, self).to(device)

