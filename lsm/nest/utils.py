import nest
import numpy as np


def get_spike_times(spike_rec):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [ms]
    """
    events = nest.GetStatus(spike_rec)[0]['events']
    min_idx = min(events['senders'])
    max_idx = max(events['senders'])
    spikes = []
    for i in range(min_idx, max_idx + 1):
        idx = np.where(events['senders'] == i)
        spikes.append(events['times'][idx])
    return spikes
