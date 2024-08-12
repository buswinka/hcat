import operator as op

import numpy as np
import scipy.stats
from scipy import signal, stats
from hcat.state.abr_waveform_dataclass import ABRWaveform, ABRExperiment
from typing import *

latencies = {  # the latencies of each wave somehow...
    1: stats.norm(1.5, 0.5),
    2: stats.norm(2.5, 1),
    3: stats.norm(3.0, 1),
    4: stats.norm(4.0, 1),
    5: stats.norm(5.0, 2),
}


def find_peaks(
        waveform: ABRWaveform,
        distance: float = 0.25e-3,
        prominence: float = 25,
        wlen: int | None = None,
        invert: bool = False,
        detrend: bool = False,
) -> Dict[str, Any]:
    """
    Algorithm to find the peaks of an ABR waveform...


    :param waveform: ABRWaveform...
    :param distance: Minimum horizontal distance between peaks...
    :param prominence:
    :param wlen: Window Length!
    :param invert:
    :param detrend:
    :return:
    """
    y: np.ndarray = -waveform.get_data() if invert else waveform.get_data()
    y: np.ndarray = signal.detrend(y) if detrend else y

    # time: np.ndarray = waveform.
    sample_rate: float = (
            1e6 / waveform.parent.get_sample_rate()
    )  # sample rate in seconds!
    time: np.ndarray = waveform.get_time(as_array=True)

    prominence = np.percentile(y, prominence)
    i_distance: int = round(sample_rate * distance)  # fs is the sample rate...

    wlen: int = round(sample_rate * wlen) if wlen is not None else wlen

    kwargs = {"distance": i_distance, "prominence": prominence, "wlen": wlen}
    indices, metrics = signal.find_peaks(y, **kwargs)

    metrics.pop("left_bases")
    metrics.pop("right_bases")
    metrics["x"] = time[indices]
    metrics["y"] = (-1 if invert else 1) * y[indices]
    metrics["index"] = indices

    return metrics


def guess_peaks(
        metrics: Dict[str, Any], latency: Dict[int, scipy.stats._continuous_distns.norm_gen]
) -> Dict[int, Dict[str, float]]:
    """
    Guesses peaks from a dictonary output by find_peaks()

    :param metrics:
    :param latency:
    :return:
    """
    p_score_norm = metrics["prominences"] / metrics["prominences"].sum()
    guess: Dict[int, Dict[str, float]] = {}
    for i in sorted(latency.keys()):
        l = latency[i]
        l_score = l.pdf(metrics["x"])
        l_score_norm = l_score / np.sum(l_score + 1e-8)
        score = 5 * l_score_norm + p_score_norm
        valid = False
        while not valid:
            m: int = np.argmax(score)
            x, y = metrics['x'][m], metrics['y'][m]
            _metric = {"x": metrics["x"][m], "y": metrics["y"][m]}

            if not guess:
                guess[i] = _metric
                valid = True
            else:
                for k,v in guess.items():

                    prev_x, prev_y = guess[i-1]['x'], guess[i-1]['y']

                    if x == v['x'] and y == v['y']:
                        score[m] = -9999.999999
                    elif x < prev_x:
                        score[m] = -9999

            # We set score to something small, so it likeyly isnt the argmax anymore...
            if m == np.argmax(score):
                guess[i] = _metric
                valid = True


    return guess


def generate_latencies_bound(
        guess: Dict[int, Dict[str, float]], max_time: float = 8.5, sd: float = 0.5
) -> float:
    """
    Generates a bounded latencies for notch prediction...

    :param guess:
    :param max_time:
    :param sd:
    :return:
    """

    # waves = sorted(guess.values())
    n_waves = max(k for k in guess.keys())

    latency = {}
    for lower_bound, upper_bound in zip(range(1, n_waves), range(2, n_waves+1)):
        time_lower_bound = guess[lower_bound]["x"]
        time_upper_bound = guess[upper_bound]["x"]

        b = max((time_upper_bound - time_lower_bound) / sd, time_lower_bound + 0.25)

        latency[lower_bound] = stats.truncnorm(0, b, time_lower_bound, sd)
    g = guess[n_waves]
    time = g["x"]
    b = (max_time - time) / sd
    latency[n_waves] = stats.truncnorm(0, b, time, sd)
    return latency


def generate_latencies_skewnorm(guess: Dict[str, float], skew: int = 3):
    """
    Takes a guess with keys X/Y
    and updates the latencies by some amount!!!

    This makes the latency delay happen!

    :param guess:
    :param skew:
    :return:
    """
    latencies = {}
    for w, row in guess.items():
        latencies[w] = stats.skewnorm(skew, row["x"], 0.2)

    return latencies


def guess_all_waveform_peaks(
        waveforms: List[ABRWaveform], latencies: List[Any], invert=False
) -> Dict[int, Dict[str, float]]:
    """
    As far as I can tell, the loops over all waveforms and gets one for each


    :param waveforms:
    :param latencies:
    :param invert:
    :return:
    """
    waveforms = sorted(waveforms, key=op.attrgetter("level"), reverse=True)
    guesses: Dict[int, Dict[str, float]] = {}
    for w in waveforms:
        metrics = find_peaks(w, invert=invert)
        guesses[w.level]: Dict[int, Dict[str, float]] = guess_peaks(metrics, latencies)

        latencies = generate_latencies_skewnorm(guesses[w.level])

    for pn, guess in guesses[w.level].items():
        if guess['x'] < 1.0:
            raise RuntimeError
    return guesses


def guess(waveforms: List[ABRWaveform], latencies: List[Any], invert=False):
    guesses = {}
    for w in waveforms:
        metrics = find_peaks(w, invert=invert)
        guesses[w.level] = guess_peaks(metrics, latencies[w.level])
    return guesses


"""
    def guess_n(self):
        n_latencies = {}
        for w in self.waveforms:
            g = {p.wave_number: p.x for p in w.points.values() if p.is_peak()}
            g = pd.DataFrame({'x': g})
            n_latencies[w.level] = generate_latencies_bound(g)
        level_guesses = guess(self.waveforms, n_latencies, invert=True)
        self._set_points(level_guesses, Point.VALLEY)

"""


def guess_abr_peaks(
        experiment: ABRExperiment, latencies: Dict[int, Any] | None = None
) -> Any:
    if latencies is None:
        latencies = {  # the latencies of each wave somehow...
            1: stats.norm(1.5, 0.5),
            2: stats.norm(2.5, 0.5),
            3: stats.norm(3.0, 1),
            4: stats.norm(4.0, 1),
            5: stats.norm(5.0, 2),
        }
    guesses = guess_all_waveform_peaks(experiment.get_waveforms(), latencies)
    return guesses


def guess_abr_notches(experiment: ABRExperiment, peak_guesses: Dict[int, Dict[int, Dict[str, float]]]) -> Any:

    waveforms = sorted(experiment.get_waveforms(), key=op.attrgetter("level"), reverse=True)
    guesses: Dict[int, Dict[str, float]] = {}

    for w in waveforms:
        guess = peak_guesses[w.level]
        metrics = find_peaks(w, invert=True)
        latencies = generate_latencies_bound(guess)
        guesses[w.level] = guess_peaks(metrics, latencies)

    return guesses


if __name__ == "__main__":
    from hcat.cabr.EPL_parser import parse_abr_file
    import matplotlib.pyplot as plt

    experiment = parse_abr_file(
        "/Users/chrisbuswinka/Documents/Projects/hcat/test_abr_data/ABR-72-2"
    )[0]

    experiment.set_filter_params(300, 3000, 1)

    waveform: ABRWaveform = experiment.get_waveforms()[0]

    # Guesses is a dict with levels as keys...
    # {40: {1: {'x': 0.12341, 'y': 0.2412512}, 2: {...}}. }

    peaks = guess_abr_peaks(experiment)
    notches = guess_abr_notches(experiment, peaks)

    experiment.set_peaks(peaks)
    experiment.set_notches(notches)

    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    plt.figure()
    for i, waveform in enumerate(experiment.get_waveforms()):
        i = len(experiment) - i
        time = waveform.get_time(as_array=True)
        data = waveform.get_data()
        level = waveform.get_level()

        data = data[time < 8.0]
        time = time[time < 8.0]

        plt.plot(time, data + i, 'k')

        for peak_number, guess in peaks[level].items():
            plt.plot(guess["x"], guess["y"] + i, "x", c=default_colors[peak_number-1])

        for peak_number, guess in notches[level].items():
            plt.plot(guess["x"], guess["y"] + i, "o", c=default_colors[peak_number-1])

    plt.show()
