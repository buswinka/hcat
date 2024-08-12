from __future__ import annotations

from abc import ABC

import torch
from hcat.state.abr_waveform_dataclass import *
from hcat.lib.types import *
import os.path
import hcat.cabr
from torch import Tensor
from typing import *
import numpy as np
import warnings


class TempStore:
    def __init__(self):
        self.store: Dict[str, ExperimentDict] = {}

    def add_experiment(
        self,
        experiments: List[ABRExperiment] | ABRExperiment,
    ) -> TempStore | None:
        """
        Adds an experiment to the store. Autosaves after each addition by default.

        :param experiments: A single, or list, of ABRExperiment to add to the store.
        :param autosave: Re-write the store file after each addition...
        :param overwrite: Overwrite the file if previously added.
        :return: None
        """

        if isinstance(experiments, ABRExperiment):
            experiments = [experiments]
        elif not isinstance(experiments, List):
            raise RuntimeError(
                "Only a single, or list of multiple, ABRExperiments may be added to the store."
            )

        for experiment in experiments:
            levels: List[int] = experiment.get_levels()
            threshold: int = experiment.get_threshold()
            frequency: float = experiment.get_frequency()
            waveform_mat: Tensor = self._waveform_mat_from_experiment(experiment)
            filepath: str = experiment.get_filepath()
            all_above_thr = experiment.all_above_threshold
            all_below_thr = experiment.all_below_threshold
            filter_params: FilterParams = experiment.get_filter_params()
            peaks: ABRPeaks = experiment.get_peaks()
            notches: ABRPeaks = experiment.get_notches()

            if all_above_thr:
                thr_str = "ALL_ABOVE"
            elif all_below_thr:
                thr_str = "ALL_BELOW"
            else:
                thr_str = threshold

            key = filepath + str(frequency) + self.filter_to_key(filter_params)

            self.store[key]: ExperimentDict = {
                "waveforms": waveform_mat,
                "levels": levels,
                "threshold": thr_str,
                "frequency": frequency,
                "filter_params": experiment.get_filter_params(),
                "peaks": peaks,
                "notches": notches
            }

        return self

    def __contains__(self, experiment: ABRExperiment) -> bool:
        """is an experimetn in the"""
        if not isinstance(experiment, ABRExperiment):
            return False
        filepath = (
            experiment.filepath
            + str(experiment.frequency)
            + self.filter_to_key(experiment.get_filter_params())
        )
        return filepath in self.store

    def __len__(self) -> int:
        return len(self.store.keys())

    def __getitem__(self, item: str | ABRExperiment):
        if isinstance(item, str):
            return self.store[item]
        elif isinstance(item, ABRExperiment):
            key = item.filepath + str(item.frequency) + self.filter_to_key(item.get_filter_params())
            return self.store[key]
        else:
            raise KeyError("item is not in store.")

    def get(self, key):
        return self.store[key]

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    @staticmethod
    def filter_to_key(filter_params: FilterParams | None) -> str:
        if not filter_params:
            return "None"

        f0 = filter_params["f0"]
        f1 = filter_params["f1"]
        order = filter_params["order"]
        return f"{f0}_{f1}_{order}"

    @staticmethod
    def _waveform_mat_from_experiment(experiment: ABRExperiment) -> Tensor:
        data = [w.data for w in experiment.get_waveforms()]
        data = torch.tensor(np.array(data))
        data = data.unsqueeze(0).unsqueeze(0).float()
        data = data[..., 0:256]
        return data


class ABRStore:
    """
    Creates a persistent ABR store. This is stored in a file which is loaded and keeps track of all
    ABR data analyzed by the software. The idea is: abr data is pretty lightweight. You can keep track of
    thousands of them for basically nothing. Keeping track of all data then lets one train a deep
    neural network without the user worrying about where the training data is...
    """

    def __init__(self):
        """
        Loads the persistent store if available. If not, creates one...
        """
        self.store_file_location = os.path.join(
            hcat.cabr.__path__[0], "abr_store.trch"
        )

        self.store: Store = (
            torch.load(self.store_file_location)
            if os.path.exists(self.store_file_location)
            else {}
        )

    def add_experiment(
        self,
        experiments: List[ABRExperiment] | ABRExperiment,
        autosave: bool = True,
        overwrite: bool = True,
    ) -> ABRStore | None:
        """
        Adds an experiment to the store. Autosaves after each addition by default.

        :param experiments: A single, or list, of ABRExperiment to add to the store.
        :param autosave: Re-write the store file after each addition...
        :param overwrite: Overwrite the file if previously added.
        :return: None
        """

        if isinstance(experiments, ABRExperiment):
            experiments = [experiments]
        elif not isinstance(experiments, List):
            raise RuntimeError(
                "Only a single, or list of multiple, ABRExperiments may be added to the store."
            )

        for experiment in experiments:
            levels: List[int] = experiment.get_levels()
            threshold: int = experiment.get_threshold()
            frequency: float = experiment.get_frequency()
            waveform_mat: Tensor = self._waveform_mat_from_experiment(experiment)
            filepath: str = experiment.get_filepath()
            all_above_thr = experiment.all_above_threshold
            all_below_thr = experiment.all_below_threshold
            peaks: ABRPeaks = experiment.get_peaks()
            notches: ABRPeaks = experiment.get_notches()

            if all_above_thr:
                thr_str = "ALL_ABOVE"
            elif all_below_thr:
                thr_str = "ALL_BELOW"
            else:
                thr_str = threshold

            if threshold is None and not all_above_thr and not all_below_thr:
                raise RuntimeError(
                    "Threshold must be set before adding to ABR Store..."
                )
            if filepath in self.store and not overwrite:
                warnings.warn(
                    "The ABR file has been previously written to the store and has been skipped. "
                    "To disable this behavior, pass the keyword argument overwrite=True to this function."
                )

            # if file is already there, and the overwrite keyword is true, we dont write to store.
            if filepath in self.store and not overwrite:
                return

            self.store[filepath + str(frequency)]: ExperimentDict = {
                "waveforms": waveform_mat,
                "levels": levels,
                "threshold": thr_str,
                "frequency": frequency,
                "filter_params": experiment.get_filter_params(),
                "peaks": peaks,
                "notches": notches
            }

        if autosave:
            torch.save(self.store, self.store_file_location)

        return self

    def __contains__(self, experiment: ABRExperiment) -> bool:
        """is an experimetn in the"""
        if not isinstance(experiment, ABRExperiment):
            return False
        filepath = experiment.filepath + str(experiment.frequency)
        return filepath in self.store

    def save(self) -> ABRStore:
        """Saves the store to disk"""
        torch.save(self.store, self.store_file_location)
        return self

    def save_as(self, filepath: str) -> ABRStore:
        """Saves the store to a user set filename"""
        torch.save(self.store, filepath)
        return self

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: str | ABRExperiment):
        if isinstance(item, str):
            return self.store[item]
        elif isinstance(item, ABRExperiment):
            key = item.filepath + str(item.frequency)
            return self.store[key]
        else:
            raise KeyError("item is not in store.")

    def keys(self):
        return self.store.keys()

    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    @staticmethod
    def _waveform_mat_from_experiment(experiment: ABRExperiment) -> Tensor:
        data = [w.data for w in experiment.get_waveforms()]
        data = torch.tensor(np.array(data))
        data = data.unsqueeze(0).unsqueeze(0).float()
        data = data[..., 0:256]
        return data
