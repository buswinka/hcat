import torch
from torch import Tensor
import torch.optim
from torch.utils.data import DataLoader, Dataset
from hcat.cabr.abr_store import ABRStore
from typing import *


class ABRStoreLoader(Dataset):
    """minimalistic dataloader from an ABRStore"""

    def __init__(self, transforms: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = lambda x: x):
        super(ABRStoreLoader, self).__init__()

        self.store = ABRStore()

        self.data_dicts: List[Dict[str, Tensor | int | List[int] | float]] = [
            dd for dd in self.store.values() if dd["waveforms"].shape[2] <= 10
        ]
        """
        KEYS:
            'waveforms':Tensor
            'levels':List[int]
            'threshold':int
            'frequency':float
        """

        self.data_dicts: List[Dict[str, Tensor]] = self._to_tensor()  # DO THIS FIRST...
        self.data_dicts: List[Dict[str, Tensor]] = self._calculate_gt_index_from_levels()

        for i, d in enumerate(self.data_dicts):
            assert 'threshold_index' in d, f'{d.keys()=}, {i=}, {d["threshold"]=}, {d["levels"]=}'

        self.transforms = transforms

    def _calculate_gt_index_from_levels(self) -> List[Dict[str, Tensor]]:
        """ calculates the gt index from the given threshold and levels... """
        for d in self.data_dicts:
            thr: Tensor = d['threshold']

            for j, l in enumerate(d['levels']):
                if l == thr:
                    d['threshold_index'] = torch.tensor(j, device=thr.device)
                    break

            if 'threshold_index' not in d:
                if thr == -1:
                    d['threshold_index'] = torch.tensor(-1, device=thr.device)
                elif thr == -2:
                    d['threshold_index'] = torch.tensor(-2, device=thr.device)
                else:
                    raise RuntimeError(f'{d=}')

            assert 'threshold_index' in d, f'{l=}, {d["threshold"]}, {d["levels"]}'
            # self.data_dicts[i] = d
        return self.data_dicts

    def _to_tensor(self) -> List[Dict[str, Tensor]]:
        for i, d in enumerate(self.data_dicts):
            d["levels"] = torch.tensor(d["levels"])
            if isinstance(d['threshold'], int):
                d["threshold"] = torch.tensor(d["threshold"])
            elif d['threshold'] == 'NO_RESPONSE':
                d['threshold'] = torch.tensor(-1)
            elif d['threshold'] == 'ALL_ABOVE':
                d['threshold'] = torch.tensor(-2)

            d["frequency"] = torch.tensor(d["frequency"])
            self.data_dicts[i] = d
        return self.data_dicts

    def to(self, device: str | torch.device) -> Dataset:
        for i, d in enumerate(self.data_dicts):
            d["waveforms"] = d["waveforms"].to(device)
            d["levels"] = d["levels"].to(device)
            d["threshold"] = d["threshold"].to(device)
            d["frequency"] = d["frequency"].to(device)
            d["threshold_index"] = d["threshold_index"].to(device)

            self.data_dicts[i] = d
        return self

    def cuda(self) -> Dataset:
        return self.to("cuda:0")

    def cpu(self) -> Dataset:
        return self.to("cpu")

    def get_class_adjustment_weights(self, size: int = 22, device: str = 'cpu') -> Tensor:
        indicies = [d['threshold_index'] for d in self.data_dicts]

        # Run a rough histogram to caluclate the max values
        hist: Dict[int, int] = {}
        for ind in indicies:
            if ind.item() not in hist:
                hist[ind.item()] = sum(ind == i for i in indicies)
        max_n_occurrence = max(v for k, v in hist.items())

        # Weight matrix scales the output based on the relative number of each label.
        # A label with 30 occurrences should be roughly 30x LESS influential than one label...
        weight_matrix = torch.ones(size, device=device) * max_n_occurrence
        for index, values in hist.items():
            weight_matrix[index] = weight_matrix[index] / values

        return weight_matrix






    def __len__(self) -> int:
        return len(self.data_dicts)

    def __getitem__(self, item) -> Dict[str, Tensor]:
        return self.transforms(self.data_dicts[item])


class MultiDataset(Dataset):
    def __init__(self, *args):
        r"""
        A utility class for joining multiple datasets into one accessible class. Sometimes, you may subdivide your
        training data based on some criteria. The most common is size: data from folder data/train/train_alot must be sampled 100 times
        per epoch, while data from folder data/train/train_notsomuch might only want to be sampled 1 times per epoch.

        You could construct a two skoots.train.dataloader.dataset objects for each
        and access both in a single MultiDataset class...

        :param args:
        :type args:
        """
        self.datasets: List[Dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]

        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self):
        return len(self._mapped_indicies)

    def __getitem__(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offset
        try:
            return self.datasets[i][item - _offset]
        except Exception:
            print(i, _offset, item - _offset, item, len(self.datasets[i]))
            raise RuntimeError

    def to(self, device: str):
        """
        Sends all data stored in the dataloader to a device. Occurs for ALL wrapped datasets.

        :param device: torch device for images, masks, and skeletons
        :return: self
        """
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self):
        """alias for self.to('cuda:0')"""
        for i in range(self.num_datasets):
            self.datasets[i].to("cuda:0")
        return self

    def cpu(self):
        """alias for self.to('cpu')"""
        for i in range(self.num_datasets):
            self.datasets[i].to("cpu")
        return self



def colate(x: List[Dict[str, Tensor]]):
    data_stack = []
    gt_stack = []
    for data_dict in x:
        data = data_dict["waveforms"]
        if data.shape[2] < 20:
            more_c = 20 - data.shape[2]
            zeros = torch.zeros((1, 1, more_c, data.shape[-1]), device=data.device)
            data = torch.concat((data, zeros), dim=2)
            data = data[..., 0:256]
        elif data.shape[2] > 20:
            print(data.shape)

        data_stack.append(data)

        index = data_dict["threshold_index"].unsqueeze(0)
        if index < 0:
            index[0] = 22 + index
        gt_stack.append(data_dict["threshold_index"].unsqueeze(0))

    data_stack = torch.concatenate(data_stack, dim=0)
    gt_stack = torch.concatenate(gt_stack, dim=0)

    return data_stack, gt_stack


if __name__ == "__main__":
    print(len(ABRStoreLoader().to("cpu")))
