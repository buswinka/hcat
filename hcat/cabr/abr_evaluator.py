import os.path

from hcat.cabr.abr_predictor_model import DeepABRModel
import torch
from torch import Tensor
import os.path

from typing import *
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment


class calculate_threshold:
    def __init__(self, model_path: str | None = '/Users/chrisbuswinka/Documents/Projects/hcat/hcat/cabr/abr_picker_state_dict.trch') -> Tuple[float, str, str]:
        """
        Function which returns the likely threshold of an ABRExperiment from a DeepABR Model

        :param experiment:
        :param criterion:
        :return:
        """
        self.model_path = model_path
        self.model = DeepABRModel()

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

        if model_path:
            self.load_model_weights(model_path)

    def load_model_weights(self, path: str):
        """sets the model from"""
        if os.path.exists(path):
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict, strict=False)
            self.model_path = path

    @staticmethod
    def _argmax(vals: List[float]) -> int:
        index, max_val = -1, -float("inf")
        for i, val in enumerate(vals):
            if val > max_val:
                index, max_val = i, val
        return index

    def __call__(self, experiment: ABRExperiment) -> Tuple[int| str, List[float], None]:
        """
        Evaluates the deepABR model

        :param experiment:
        :return:
        """
        levels = [w.level for w in experiment.get_waveforms()]
        thr_probabilities: List[float] = evaluate_model(experiment, self.model)

        argmax = self._argmax(thr_probabilities)

        if argmax == len(thr_probabilities) - 3:  # ALL_ABOVE
            return 'ALL_ABOVE', thr_probabilities, None

        elif argmax == len(thr_probabilities) - 2:  # ALL_BELOW
            return 'ALL_BELOW', thr_probabilities, None
        else:
            thr_level = levels[argmax]
            return thr_level, thr_probabilities, None



def evaluate_model(experiment: ABRExperiment, model: DeepABRModel) -> List[float]:
    """
    Evaluates a deep ABR model on an ABR Experiment and returns the probability of threshold for each waveform.

    :param experiment: Generic ABR experiment with less than 20 Waveforms
    :param model: pretrained DeepABRModel
    :return: List[float] probabilities for each waveform.
    """
    device = model.device
    data: Tensor = torch.stack(
        [
            torch.tensor(w.data).half().to(device)
            for w in experiment.get_waveforms()
        ],
        dim=0,
    ).unsqueeze(0).unsqueeze(0)
    # Shape[1, 1, N, M]

    N_waveforms: int = data.shape[2]

    # DeepABRModel expects a Tensor with Shape [B=1, C=1, N=20, C=256]
    data = data[..., 0:256]
    if data.shape[2] < 20:  # Model expects max of 20 levels. MORE THAN ENOUGH!
        more_c: int = 20 - data.shape[2]
        zeros = torch.zeros((1, 1, more_c, data.shape[-1]), device=data.device)
        data = torch.concat((data, zeros), dim=2)

    model = model.float().eval()
    out: Tensor = model(data.float())  # Tensor[]
    out: List[float] = out.squeeze().cpu().tolist()
    out: List[float] = out[0:N_waveforms] + out[-2::]

    return out
