import os.path
import re
from copy import copy
from typing import List
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.signal
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
import glob

from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment


def parse_abr_file(filepath: str) -> ABRExperiment:
    """
    parse an ABR waveform file from EPL

    :param filepath:
    :return:
    """
    with open(filepath, "rb") as file:
        leveltext = file.read()

    analyzed_file = glob.glob(os.path.join(filepath, '*-analyzed.txt'))
    if not analyzed_file:
        return
    else:
        with open(analyzed_file[0], 'rb') as file:
            analyzed_text = file.read()
            threshold: List[str] | None = re.search(f"Threshold \(dB SPL\)\: (\d\d;){1,}.(\d){1,}")
            threshold: List[str] | None = re.findall(r"[-+]?(?:\d*\.*\d+)", str(threshold))
            threshold: float | None = float(threshold[0]) if threshold else None


    # Have to convert to list then back to np array or it breaks. I dont know why.
    levelheader = ":LEVELS:"
    levels: List[str] | None = re.search(r"\:LEVELS\:(\d\d;){1,}", str(leveltext))
    if levels is None:
        raise RuntimeError("Could not determine the stimulus levels from ABR file")

    levels: str = levels[0][len(levelheader) : -1 : 1]
    levels: np.array = np.fromstring(levels, dtype=int, sep=";")
    levels: List[float] = levels.tolist()

    freq: List[str] | None = re.search("FREQ\:\ \ ?(\d){1,}.(\d){1,}", str(leveltext))
    freq: float | None = float(freq[0][6:-1:1]) if freq else None

    if freq is None:
        raise RuntimeError("Could not determine the frequency from ABR file")

    sample_rate: List[str] | None = re.search("SAMPLE \(.?sec\)\:\ \ ?(\d){1,}.((\d){1,})?", str(leveltext))
    sample_rate: List[str] | None = re.findall(r"[-+]?(?:\d*\.*\d+)", str(sample_rate))
    sample_rate: float | None = float(sample_rate[0]) if sample_rate else None


    experiment = ABRExperiment(filename=filepath,
                               frequency=freq,
                               sample_rate=sample_rate)

    _data: pd.DataFrame = pd.read_csv(
        filepath,
        delimiter="\t",
        names=levels,
        header=10,
        engine="python",
        encoding="unicode_escape",
    )

    for level in levels:
        waveform = ABRWaveform(
            level=level,
            data=_data[level].to_numpy(),
            dt=1.0,
            parent=experiment
        )
        experiment.add_waveform(waveform)

    data = [w.data for w in experiment.get_waveforms()]
    data = torch.tensor(np.array(data))

    if data.shape[0] < 10:
        more_c = 10 - data.shape[0]
        zeros = torch.zeros((more_c, data.shape[1]))
        data = torch.concat((data, zeros), dim=0)

    data = data.unsqueeze(0).unsqueeze(0).float()
    data = data[..., 0:256]

    for i, w in enumerate(experiment.get_waveforms()):
        if w == threshold:
            break

    return data, i

