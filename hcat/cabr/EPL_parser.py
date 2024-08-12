import os.path
import re
from copy import copy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.signal
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment


def parse_abr_file(filepath: str) -> ABRExperiment:
    """
    parse an ABR waveform file from EPL

    :param filepath:
    :return:
    """
    with open(filepath, "r", encoding="ISO-8859-1") as file:
        leveltext = file.read()

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

    sample_rate: List[str] | None = float(re.search("SAMPLE \(.sec\): ([0-9]+)", str(leveltext)).group(1))
    # sample_rate: List[str] | None = re.findall(r"[-+]?(?:\d*\.*\d+)", str(sample_rate))
    # sample_rate: float | None = float(sample_rate[0]) if sample_rate else None
    # sample rate is assumed to be uSec

    experiment = ABRExperiment(filepath=filepath,
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

    return [experiment]


class old_Parser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.id = filepath

        levelheader = ":LEVELS:"

        with open(filepath, "r", encoding="ISO-8859-1") as file:
            leveltext = file.read()

        # Have to convert to list then back to np array or it breaks. I dont know why.
        levels: List[str] | None = re.search(r"\:LEVELS\:(\d\d;){1,}", str(leveltext))
        if levels is None:
            raise RuntimeError("Could not determine the stimulus levels from ABR file")

        levels: str = levels[0][len(levelheader) : -1 : 1]
        levels: np.array = np.fromstring(levels, dtype=int, sep=";")
        levels: List[float] = levels.tolist()

        # Gets the levels out of the class
        self._levels = copy(levels)
        self.levels: np.array = np.array(levels)

        # ASSUME uSec
        freq: List[str] | None = re.search(
            "FREQ\:\ \ ?(\d){1,}.(\d){1,}", str(leveltext)
        )
        freq: float | None = float(freq[0][6:-1:1]) if freq else None

        if freq is None:
            raise RuntimeError("Could not determine the frequency from ABR file")

        freq = np.floor(freq)
        self.frequency: np.array = freq

        self._data: pd.DataFrame = pd.read_csv(
            self.filepath,
            delimiter="\t",
            names=levels,
            header=10,
            engine="python",
            encoding="unicode_escape",
        )

        self.threshold, self.case, self.flag = self.calculate_threshold()

    def __getitem__(self, item):
        if item not in self._levels:
            pass
            # raise IndexError(f'{item=} not in ABR experiment')

        return np.array(self._data[item].to_list())

    def keys(self):
        return self._levels

    def values(self):
        return [np.array(self._data[k].to_list()) for k in self._levels]

    def items(self):
        """not a generator but works the same way..."""
        return [(k, v) for k, v in zip(self.keys(), self.values())]

    def calculate_threshold(self):
        if "criterion" not in globals():
            criterion = 0.35

        levels, corrmat = self._calculate_corr_level_function_chris_version()

        def sigmoid(x, a, b, c, d):
            return a + (b - a) / (1 + (10 ** (d * (c - x))))

        def power(x, a, b, c):
            return a * x**b + c

        def inverse_power(y, coeff):
            return ((y - coeff[2]) / coeff[0]) ** (1 / coeff[1])

        def inverse_sigmoid(y, coeff):
            return (
                coeff[2]
                - np.log10(((coeff[1] - coeff[0]) / (y - coeff[0])) - 1) / coeff[3]
            )

        try:
            sig_coeff, _ = scipy.optimize.curve_fit(
                sigmoid,
                levels,
                corrmat,
                p0=[
                    np.min(corrmat),
                    np.max(corrmat),
                    np.max(levels) / 2,
                    1 / np.max(levels),
                ],
                maxfev=10000,
            )
        except:
            Warning("Could not converge on sigmoid Fit: Manually setting sig_coeff")
            sig_coeff = [100, 100, 100, 100]

        try:
            pow_coeff, _ = scipy.optimize.curve_fit(
                power, levels, corrmat, p0=[1 / 70, 1, 0.1], maxfev=10000
            )
        except:
            Warning("Could not converge on power Fit: Manually setting sig_coeff")
            pow_coeff = [100, 100, 100]
            # Handles Fringe Case when scipy.optimize.curve_fit cant hande the fit, manually sets coeff
            # to something unreasable so that pow will have a low r2 and never be used for threshold determination

        RMS_sig = mean_squared_error(
            corrmat,
            sigmoid(levels, sig_coeff[0], sig_coeff[1], sig_coeff[2], sig_coeff[3]),
        )
        RMS_pow = mean_squared_error(
            corrmat, power(levels, pow_coeff[0], pow_coeff[1], pow_coeff[2])
        )

        pow_r2 = sklearn.metrics.r2_score(
            corrmat, power(levels, pow_coeff[0], pow_coeff[1], pow_coeff[2])
        )

        C1 = sig_coeff[0] < criterion and sig_coeff[1] > criterion
        C1 = C1 and sig_coeff[3] > 0.005 and sig_coeff[3] < 1
        C2 = RMS_sig < RMS_pow
        C3 = pow_r2 > 0.7
        C4 = np.max(corrmat) > criterion
        threshold = np.max(corrmat)
        FLAG = "clear"
        CASE = "---"

        # Decision tree for threshold
        if C1:
            if C2:
                threshold = inverse_sigmoid(criterion, sig_coeff)
                CASE = "A"
            else:
                if C3:
                    threshold = inverse_power(criterion, pow_coeff)
                    CASE = "B"
                else:
                    if C4:
                        threshold = inverse_power(criterion, pow_coeff)
                        FLAG = "NOISY"
                        CASE = "C"

                    else:
                        "D: Visual"
                        FLAG = "COULD NOT BE DETERMINED"
                        CASE = "D"
        else:  # if C1
            if C3:
                threshold = inverse_power(criterion, pow_coeff)
                CASE = "B"
            else:
                if C4:
                    threshold = inverse_power(criterion, pow_coeff)
                    CASE = "C"
                    FLAG = "NOISY"
                else:
                    FLAG = "COULD NOT BE DETERMINED"
                    CASE = "D"

        badthreshold = isinstance(threshold, complex)
        badthreshold = badthreshold or np.isnan(threshold)
        badthreshold = badthreshold or int(threshold) < 1 or int(threshold) > 100

        if badthreshold:
            threshold = None
            print("Threshold could not be determined")

        return threshold, CASE, FLAG

    @staticmethod
    def _xcov(x: np.array, y: np.array):
        """
        Reimplementation of MATLAB's xcov function
        x, y must be of equal length

        :param x: singal 1
        :param y: signal 2
        :return: xcov of x and y
        """
        x = x - np.mean(x)
        y = y - np.mean(y)

        x_autocor = np.correlate(x, x, "full")
        x_autocor = x_autocor[len(x_autocor) // 2]

        y_autocor = np.correlate(y, y, "full")
        y_autocor = y_autocor[len(y_autocor) // 2]

        corr = scipy.signal.correlate(x, y)

        corr = corr / np.sqrt((x_autocor * y_autocor))

        return corr

    def _calculate_corr_level_function(self):
        len_datamat = len(self._data.columns)

        corrmat = np.zeros(len_datamat - 1)
        levels = np.zeros(len_datamat - 1)

        for i, level in enumerate(self._data.columns):
            if i == len_datamat - 1:
                break

            # Pull in unfiltered data from txt file
            x = self._data[level]
            y = self._data[self._data.columns[i + 1]]

            # Construct a filter as per Kirupa's paper
            # Note, they used a zero-pole filter, while Im using a sos filter
            # It should be the same
            filter = scipy.signal.butter(
                4, (200, 10000), fs=1 / (40 * 10**-6), btype="bandpass", output="sos"
            )

            x = scipy.signal.sosfilt(filter, x)
            y = scipy.signal.sosfilt(filter, y)

            # # Some Issues with Kirupa Code! Excluding last half of ABR Waveform!!!!!!
            # ######
            x = x[0 : len(x) // 3 : 1]
            y = y[0 : len(y) // 3 : 1]
            # ######
            # #

            # Python implementation of matlab's xcov function
            corr = self._xcov(x, y)

            # Only want correlation at lag 0, which is in middle of matrix
            corrmat[i] = corr[len(corr) // 2]

            # # CHRIS TWEAK TO ALGORITHM: due to time lag, in abr, we take max autocorr over a range
            # #
            # corrmat[i] = np.max(corr[len(corr)//2 - 30 : len(corr)//2 + 30: 1])
            # #
            # #

            levels[i] = float(level)

        return levels, corrmat

    def _calculate_corr_level_function_chris_version(self):
        filter = scipy.signal.butter(
            4, (200, 10000), fs=1 / (40 * 10**-6), btype="bandpass", output="sos"
        )
        len_datamat = len(self._data.columns)

        corrmat = np.zeros(len_datamat)
        levels = np.zeros(len_datamat)

        for i, lev in enumerate(self._data.columns):
            levels[i] = float(lev)
            corr = []

            for j, compare_level in enumerate(self._data.columns):
                if lev != compare_level:
                    x = self._data[lev]
                    y = self._data[compare_level]

                    x = scipy.signal.sosfilt(filter, x)
                    y = scipy.signal.sosfilt(filter, y)

                    x = x[0 : len(x) // 1 : 1]
                    y = y[0 : len(y) // 1 : 1]

                    autocorrelation = self._xcov(x, y)
                    corr.append(
                        np.max(
                            autocorrelation[
                                len(autocorrelation) // 2
                                - 30 : len(autocorrelation) // 2
                                + 30 : 1
                            ]
                        )
                    )

            corrmat[i] = np.mean(corr)

        return levels, corrmat

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, filename):
        regex = r"ABR-(\d){1,}-"
        filename = os.path.basename(self.filepath)
        id = re.search(regex, filename)
        id = re.search(r"(\d){1,}", id[0])

        self._id = id[0]

    def diagnostics(self):
        """
        Diagnostics on the parser thresholding algorithm
        :return:
        """
        print(self.filepath)
        print("Threshold: ", self.threshold)
        print("CASE: ", self.case)
        print("FLAG: ", self.flag)

        return self.case, self.flag

    def save_figure(self, savepath):
        """
        Save a figure of the abr waveforms to a seperate folder denoted by savepath
        :param savepath:
        :return:
        """
        fig, ax = plt.subplots(figsize=(4, 5))
        offset = np.logspace(0, 7.5, len(self.levels), base=1.3)

        for i, value in enumerate(self._data.columns):
            waveform = self._data[value]
            length = len(waveform)
            time = np.linspace(0, len(waveform) * 40 * 10**-6, len(waveform)) * 1000

            bandpass_filter = scipy.signal.butter(
                4, (200, 10000), fs=1 / (40 * 10**-6), btype="bandpass", output="sos"
            )

            waveform = scipy.signal.sosfilt(bandpass_filter, waveform)

            waveform = waveform + offset[i]

            ax.plot(time[1 : length // 2 : 1], waveform[1 : length // 2 : 1], c="k")

        plt.yticks(offset, self._data.columns)
        plt.xlabel("Time (ms)")
        plt.ylabel("Level (dB)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        name = self.id + "-" + str(self.frequency) + ".jpeg"

        plt.savefig(savepath + name)
        plt.close()


if __name__ == "__main__":
    p = parse_abr_file("/Users/chrisbuswinka/Documents/Projects/hcat/test_abr_data/ABR-72-1")
    # for k, v in p.items():
    #     print(k, v)
    # print(p._data)
    # p.save_figure('../../test_abr_data/')
