import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from typing import *
import numpy as np
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from hcat.cabr.EPL_parser import parse_abr_file
from hcat.cabr.abr_dataloader import ABRStoreLoader
from hcat.cabr.abr_transforms import merged_transform
import torch.optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    epochs: trange,
) -> Tuple[nn.Module, DataLoader, Optimizer, nn.CrossEntropyLoss, trange, float]:
    avg_loss = []
    for data, target in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

    # if i % 100 == 0:
    #     print(f'\n{target.tolist()=}\n{out.tolist()=}\n{loss.item()=}')
    avg_loss = torch.tensor(avg_loss).mean().item()

    epochs.desc = f"{avg_loss:0.8f}"
    return model, dataloader, optimizer, criterion, epochs, avg_loss
