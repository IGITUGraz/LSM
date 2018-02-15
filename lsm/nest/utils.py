import nest
import numpy as np


def get_spike_times(spike_rec, rec_nodes):
    """
       Takes a spike recorder spike_rec and returns the spikes in a list of numpy arrays.
       Each array has all spike times of one sender (neuron) in units of [ms]
    """
    events = nest.GetStatus(spike_rec)[0]['events']
    spikes = []
    for i in rec_nodes:
        idx = np.where(events['senders'] == i)
        spikes.append(events['times'][idx])
    return spikes
