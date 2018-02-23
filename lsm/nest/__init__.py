import nest
import numpy as np
import pylab

from lsm.nest.utils import get_spike_times
from lsm.utils import windowed_events


def create_iaf_psc_exp(n_E, n_I):
    nodes = nest.Create('iaf_psc_exp', n_E + n_I,
                        {'C_m': 30.0,  # 1.0,
                         'tau_m': 30.0,  # Membrane time constant in ms
                         'E_L': 0.0,
                         'V_th': 15.0,  # Spike threshold in mV
                         'tau_syn_ex': 3.0,
                         'tau_syn_in': 2.0,
                         'V_reset': 13.8})

    nest.SetStatus(nodes, [{'I_e': 14.5} for _ in nodes])
    # nest.SetStatus(nodes, [{'I_e': np.minimum(14.9, np.maximum(0, np.random.lognormal(2.65, 0.025)))} for _ in nodes])

    return nodes[:n_E], nodes[n_E:]


def connect_tsodyks(nodes_E, nodes_I):
    f0 = 10.0

    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)
    n_syn_exc = 2  # number of excitatory synapses per neuron
    n_syn_inh = 1  # number of inhibitory synapses per neuron

    w_scale = 10.0
    J_EE = w_scale * 5.0  # strength of E->E synapses [pA]
    J_EI = w_scale * 25.0  # strength of E->I synapses [pA]
    J_IE = w_scale * -20.0  # strength of inhibitory synapses [pA]
    J_II = w_scale * -20.0  # strength of inhibitory synapses [pA]

    def get_u_0(U, D, F):
        return U / (1 - (1 - U) * np.exp(-1 / (f0 * F)))

    def get_x_0(U, D, F):
        return (1 - np.exp(-1 / (f0 * D))) / (1 - (1 - get_u_0(U, D, F)) * np.exp(-1 / (f0 * D)))

    def gen_syn_param(tau_psc, tau_fac, tau_rec, U):
        return {"tau_psc": tau_psc,
                "tau_fac": tau_fac,  # facilitation time constant in ms
                "tau_rec": tau_rec,  # recovery time constant in ms
                "U": U,  # utilization
                "u": get_u_0(U, tau_rec, tau_fac),
                "x": get_x_0(U, tau_rec, tau_fac),
                }

    def connect(src, trg, J, n_syn, syn_param):
        nest.Connect(src, trg,
                     {'rule': 'fixed_indegree', 'indegree': n_syn},
                     dict({'model': 'tsodyks_synapse', 'delay': delay,
                           'weight': {"distribution": "normal_clipped", "mu": J, "sigma": 0.7 * abs(J),
                                      "low" if J >= 0 else "high": 0.
                           }},
                          **syn_param))

    connect(nodes_E, nodes_E, J_EE, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1.0, tau_rec=813., U=0.59))
    connect(nodes_E, nodes_I, J_EI, n_syn_exc, gen_syn_param(tau_psc=2.0, tau_fac=1790.0, tau_rec=399., U=0.049))
    connect(nodes_I, nodes_E, J_IE, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=376.0, tau_rec=45., U=0.016))
    connect(nodes_I, nodes_I, J_II, n_syn_inh, gen_syn_param(tau_psc=2.0, tau_fac=21.0, tau_rec=706., U=0.25))


def inject_noise(nodes_E, nodes_I):
    p_rate = 25.0  # this is used to simulate input from neurons around the populations
    J_noise = 1.0  # strength of synapses from noise input [pA]
    delay = dict(distribution='normal_clipped', mu=10., sigma=20., low=3., high=200.)

    noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

    nest.Connect(noise, nodes_E + nodes_I, syn_spec={'model': 'static_synapse',
                                                     'weight': {
                                                         'distribution': 'normal',
                                                         'mu': J_noise,
                                                         'sigma': 0.7 * J_noise
                                                     },
                                                     'delay': dict(distribution='normal_clipped',
                                                                   mu=10., sigma=20.,
                                                                   low=3., high=200.)
    })


class LSM(object):
    def __init__(self, n_exc, n_inh, n_rec,
                 create=create_iaf_psc_exp, connect=connect_tsodyks, inject_noise=inject_noise):

        neurons_exc, neurons_inh = create(n_exc, n_inh)
        connect(neurons_exc, neurons_inh)
        inject_noise(neurons_exc, neurons_inh)

        self.exc_nodes = neurons_exc
        self.inh_nodes = neurons_inh
        self.inp_nodes = neurons_exc
        self.rec_nodes = neurons_exc[:n_rec]

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_rec = n_rec

        self._rec_detector = nest.Create('spike_detector', 1)

        nest.Connect(self.rec_nodes, self._rec_detector)

    def get_states(self, times, tau):
        spike_times = get_spike_times(self._rec_detector, self.rec_nodes)
        return LSM._get_liquid_states(spike_times, times, tau)

    @staticmethod
    def compute_readout_weights(states, targets, reg_fact=0):
        """
        Train readout with linear regression
        :param states: numpy array with states[i, j], the state of neuron j in example i
        :param targets: numpy array with targets[i], while target i corresponds to example i
        :param reg_fact: regularization factor; 0 results in no regularization
        :return: numpy array with weights[j]
        """
        if reg_fact == 0:
            w = np.linalg.lstsq(states, targets)[0]
        else:
            w = np.dot(np.dot(pylab.inv(reg_fact * pylab.eye(np.size(states, 1)) + np.dot(states.T, states)),
                              states.T),
                       targets)
        return w

    @staticmethod
    def compute_prediction(states, readout_weights):
        return np.dot(states, readout_weights)

    @staticmethod
    def _get_liquid_states(spike_times, times, tau, t_window=None):
        n_neurons = np.size(spike_times, 0)
        n_times = np.size(times, 0)
        states = np.zeros((n_times, n_neurons))
        if t_window is None:
            t_window = 3 * tau
        for n, spt in enumerate(spike_times):
            # TODO state order is reversed, as windowed_events are provided in reversed order
            for i, (t, window_spikes) in enumerate(windowed_events(np.array(spt), times, t_window)):
                states[n_times - i - 1, n] = sum(np.exp(-(t - window_spikes) / tau))
        return states
