import torch
from torch import Tensor
from typing import *


def merged_transform(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    - flip x/y of the abr waveform
    - shuffle up or down
        - repeat the top or repeat the bottom a set number of times...
    - scale randomly
    - left right delay...
        - take first bit of each waveform, flip it, then use it to add delay...

    - linear shift up and down
    - add noise randomly
    -

    :param data_dict:
    :return:
    """

    # flip x/y of abr waveform
    data: Tensor = data_dict['waveforms'].clone()
    device = data.device
    index: Tensor = data_dict['threshold_index']

    if torch.rand((1,)) < 0.5:
        data = data * -1

    if torch.rand((1,)) < 0.5:
        data = data * torch.rand((1,), device=device)+1

    if torch.rand((1,)) < 0.5:
        data = data + torch.rand_like(data).div(5)

    if torch.rand((1,)) < 0.5:
        data = data + torch.rand((1,), device=device)

    data_dict = {'waveforms': data, 'threshold_index': data_dict['threshold_index'], 'levels': data_dict['levels'], 'threshold': data_dict['threshold']}

    return data_dict
