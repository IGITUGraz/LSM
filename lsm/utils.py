import numpy as np


def windowed_events(events, window_times, window_size):
    """
    Generate subsets of events which belong to given time windows.

    Assumptions:
    * events are sorted
    * window_times are sorted

    :param events: one-dimensional, sorted list of event times
    :param window_times: the upper (exclusive) boundaries of time windows
    :param window_size: the size of the windows
    :return: generator yielding (window_time, window_events)
    """
    for window_time in reversed(window_times):
        events = events[events < window_time]
        yield window_time, events[events > window_time - window_size]


def poisson_generator(rate, t_start=0.0, t_stop=1000.0, rng=None):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

    Note: t_start is always 0.0, thus all realizations are as if
    they spiked at t=0.0, though this spike is not included in the SpikeList.

    Inputs:
        rate    - the rate of the discharge (in Hz)
        t_start - the beginning of the SpikeTrain (in ms)
        t_stop  - the end of the SpikeTrain (in ms)

    Examples:
        >> gen.poisson_generator(50, 0, 1000)

    See also:
        inh_poisson_generator
    """

    if rng is None:
        rng = np.random

    # less wasteful than double length method above
    n = (t_stop - t_start) / 1000.0 * rate
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    number = int(number)
    if number > 0:
        isi = rng.exponential(1.0 / rate, number) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if i == len(spikes):
        # ISI buf overrun
        t_last = spikes[-1] + rng.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += rng.exponential(1.0 / rate, 1)[0] * 1000.0

        spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    return spikes
