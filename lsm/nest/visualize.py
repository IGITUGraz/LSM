import nest
import numpy as np
import math
import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from lsm.nest import utils


class Plotter(object):
    def __init__(self, lsm):
        self._exc_det = nest.Create('spike_detector', 1)
        self._inh_det = nest.Create('spike_detector', 1)
        nest.Connect(lsm.rec_nodes, self._exc_det)
        nest.Connect(lsm.inh_nodes, self._inh_det)

    @staticmethod
    def _matrix(arr, n_cols, pad_value=np.nan):
        # pad with nan and reshape into columns
        n_raw = arr.shape[0]
        n_pad = math.ceil(n_raw / n_cols) * n_cols
        arr = np.append(arr, [pad_value] * (n_pad - n_raw))
        arr = arr.reshape(n_pad // n_cols, n_cols)
        return arr

    def visualize(self, fn=None, stim_times=None, stim_lens=None, readout_times=None,
                  weights=None, weight_cols=None, states=None):
        if weights is not None or states is not None:
            fig, (ax_exc, ax_inh, ax_w) = plt.subplots(3, sharex=True)
        else:
            fig, (ax_exc, ax_inh) = plt.subplots(2, sharex=True)

        spikes_exc = utils.get_spike_times(self._exc_det)
        for i, st in enumerate(spikes_exc):
            ax_exc.scatter(st, np.ones_like(st) * i, marker='|', color='black', s=0.6)

        spikes_inh = utils.get_spike_times(self._inh_det)
        for i, st in enumerate(spikes_inh):
            ax_inh.scatter(st, np.ones_like(st) * i, marker='|', color='black', s=0.6)

        for ax in [ax_exc, ax_inh]:
            if stim_times is not None:
                for t in stim_times:
                    ax.axvline(t, color='blue')
                # TODO handle stim_lens
            if readout_times is not None:
                for t in readout_times:
                    ax.axvline(t, color='red')
            ax.get_yaxis().set_ticks([])
        ax_exc.set_ylabel("excitatory")
        ax_inh.set_ylabel("inhibitory")

        bar_width = 100
        bar_upper = 1000
        bar_lower = -1000

        if weights is not None:
            assert weights.ndim in [1, 2]
            if weights.ndim == 1:
                ws = itertools.repeat(weights[..., np.newaxis])
            else:
                assert np.size(weights, 1) == len(readout_times)
                ws = weights.T
            for t, w in zip(readout_times, ws):
                if weight_cols is not None:
                    w = Plotter._matrix(w, weight_cols)
                ax_w.imshow(w, extent=[t, t + bar_width, bar_lower, bar_upper])

        if states is not None:
            norm = mcolors.Normalize(states.min(), states.max())
            for t, s in zip(readout_times, states):
                if weight_cols is not None:
                    s = Plotter._matrix(s, weight_cols)
                offset = 0 if weights is None else bar_width + 30
                ax_w.imshow(s, norm=norm, extent=[t + offset, t + bar_width + offset, bar_lower, bar_upper])

        ax_w.set_xlim(0, max(readout_times))
        ax_w.set_ylim(bar_lower, bar_upper)
        ax_w.set_ylabel("weights / states")
        ax_w.get_yaxis().set_ticks([])

        fig.set_size_inches(20, 10)
        fig.tight_layout()
        if fn is None:
            plt.show()
        else:
            fig.savefig(fn)
