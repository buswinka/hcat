# modified from https://github.com/buranconsult/ABR/blob/master/abr/parsers/HDF5.py

import glob
import os.path
import tables

import numpy as np
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from typing import *


def load(filepath: str, abr_window=8.5e-3) -> List[ABRExperiment]:
    with tables.open_file(filepath) as fh:
        fs = fh.root.waveforms._v_attrs["fs"]
        cutoff: int = int(abr_window * fs)
        signal = fh.root.waveforms[:, :, :cutoff] * 1e6
        levels = fh.root.trial_log.read(field="level")
        available_frequencies = fh.root.trial_log.read(field="frequency")

        # Load all frequencies by default
        frequencies: np.ndarray = np.unique(available_frequencies)

        experiments: List[ABRExperiment] = []
        for frequency in frequencies:
            experiment = ABRExperiment(filename=filepath, frequency=frequency)
            mask = available_frequencies == frequency
            waveforms = [
                ABRWaveform(level=l, data=s, parent=experiments)
                for s, l in zip(signal[mask], levels[mask])
            ]
            for w in waveforms:
                experiment.add_waveform(w)
            experiments.append(experiment)
        return experiment

