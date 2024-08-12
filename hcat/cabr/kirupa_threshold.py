import os.path
import re
from copy import copy
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.signal
import scipy.signal
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from hcat.cabr.EPL_parser import parse_abr_file


def xcov(x: np.array, y: np.array) -> np.ndarray:
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


def corr_level_fn(experiment: ABRExperiment) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates a correlation level function for a given abr experiment

    :param experiment:
    :return:
    """
    filter = scipy.signal.butter(
        4, (200, 10000), fs=1 / (40 * 10**-6), btype="bandpass", output="sos"
    )
    n_waveforms = len(experiment)

    corrmat = np.zeros(n_waveforms)
    levels = np.zeros(n_waveforms)

    for i, (level0, waveform0) in enumerate(experiment.items()):
        corr = []
        levels[i] = float(level0)

        for level1, waveform1 in experiment.items():
            if level0 != level1:
                x: np.ndarray = scipy.signal.sosfilt(filter, waveform0.get_data())
                y: np.ndarray = scipy.signal.sosfilt(filter, waveform1.get_data())

                x = x[0 : len(x) // 1 : 1]
                y = y[0 : len(y) // 1 : 1]

                autocorrelation: np.ndarray = xcov(x, y)
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


def calculate_threshold(
    experiment: ABRExperiment, criterion: float = 0.35
) -> Tuple[float, str, str]:
    """
    For a given ABR Experiment, calculate the threshold using
    the method kirupa developed...

    :param experiment:
    :param criterion:
    :return: Tuple[Threshold: float, Case: str, Flag: str]
    """
    levels, corrmat = corr_level_fn(experiment)

    def sigmoid(x, a, b, c, d):
        with np.errstate(all='ignore'):
            return a + (b - a) / (1 + (10 ** (d * (c - x))))

    def power(x, a, b, c):
        with np.errstate(all='ignore'):
            return a * x**b + c

    def inverse_power(y, coeff):
        return ((y - coeff[2]) / coeff[0]) ** (1 / coeff[1])

    def inverse_sigmoid(y, coeff):
        return (
            coeff[2] - np.log10(((coeff[1] - coeff[0]) / (y - coeff[0])) - 1) / coeff[3]
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

    return threshold, CASE, FLAG


if __name__ == "__main__":
    p = parse_abr_file(
        "/Users/chrisbuswinka/Documents/Projects/hcat/test_abr_data/ABR-72-6"
    )
    p.plot()

    threshold, CASE, FLAG = calculate_threshold(p)
    print(threshold, CASE, FLAG)