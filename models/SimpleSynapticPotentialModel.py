import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

### STATIC PARAMETERS
Cm = 1 # uF/cm^2
J = 0.1*Cm # synaptic weight, tuned to single EPSP of ~200uV

def calculate_alpha(t, T1, T2, syn_w): 
    if np.heaviside(t, t) == 0: 
        return 0
    return (syn_w/(T2-T1))*np.heaviside(t, t)*(np.exp(-t/T2)-np.exp(-t/T1)) # post-synaptic conductance waveform

def get_ia_afferent_synaptic_potential(sim_dur, 
                                       scs_freq, 
                                       n_axons, 
                                       p_axonal_success=1, 
                                       Tw=80, 
                                       Uw=0.6,
                                       Ta=20,
                                       dt=0.025,
                                       seed=672945):
    
    """
    Predict the postsynaptic conductance after axonal and synaptic failure.
    """
    np.random.seed(seed) # set the seed so the network is always the same
    # check if previous-run exists 
    # file_path = f"data/synaptic_potentials/synaptic_potential_SCS_freq_{scs_freq}_n_axons_200_tw_{Tw}_uw_{Uw}_ta_{Ta}_p_success_{p_axonal_success}_dt_{dt}_simulation_duration_2000.pickle"
    # if os.path.exists(file_path):
    #     with open(file_path, 'rb') as f: 
    #         data = pickle.load(f)
    #         print("Synaptic potentials file already exists. Returning loaded data.")
    #         return data['conductance'][:n_axons][:int(sim_dur/dt)]

    # set up variables related to conductance
    t_vec_len = int(sim_dur/dt)
    g = np.zeros([n_axons, t_vec_len]) # postsynaptic conductance 
    T1 = 1/dt # dt steps
    T2 = 4/dt # dt steps
    
    # random synaptic weight per axon
    synaptic_shape = 1.2
    syn_ws = np.random.gamma(synaptic_shape, scale=J/synaptic_shape, size=n_axons)

    release_prob = get_ia_afferent_release_prob(sim_dur, 
                                       scs_freq, 
                                       n_axons, 
                                       p_axonal_success=p_axonal_success, 
                                       Tw=Tw, 
                                       Uw=Uw,
                                       Ta=Ta,
                                       dt=dt)
    
    for t in range(t_vec_len): 
        for j in range(n_axons): 
            g[j][t] = np.sum([calculate_alpha(t-release_t, T1, T2, syn_ws[j]) for release_t in release_prob[j]]) 

    # save data if it the longer simulation
    # data = {}
    # data['conductance'] = g
    # data['release'] = release_prob
    # if (sim_dur == 2000) & (n_axons == 200):
    #     with open(file_path, 'wb') as f: 
    #         pickle.dump(data, f)

    return g


def get_ia_afferent_release_prob(sim_dur, 
                                       scs_freq, 
                                       n_axons, 
                                       p_axonal_success=1, 
                                       Tw=80, 
                                       Uw=0.6,
                                       Ta=20,
                                       dt=0.025,
                                       delay=50): 
    """
    Function for modeling axonal and synaptic failure in response to stimulation. 

    SCS spikes constitute a "nascent spike", which may or may not travel to the MN itself, 
    either because of axonal or synaptic failure.

    """

    # adjust time-based parameters to dt 
    Tw = Tw/dt # dt steps, mean of exponentially-distributed random variable for wait time before recovery, OG: 850
    Ta = Ta/dt # dt steps, mean of normally-distributed random variable for axonal recovery

    # set up variables 
    t_vec_len = int(sim_dur/dt)
    p = np.ones([n_axons, t_vec_len]) # probability of synaptic transmission
    axon_activation = np.ones([n_axons, t_vec_len])
    release = [[] for _ in range(n_axons)] # time of release of vesicles
    scs_pulses = [1 if np.mod(x, int(1/scs_freq*1000/dt)) == 0 else 0 for x in range(int(delay/dt), t_vec_len)] # pulses from SCS
    scs_pulses = np.concatenate([np.zeros(int(delay/dt)), np.array(scs_pulses)])

    # run simulation of axonal and synaptic failure
    for t in range(t_vec_len): 
        for j in range(n_axons): 
            synapse_recover = True
            if (scs_pulses[t] == 1):      
                if (random.random() < p_axonal_success) & (axon_activation[j][t] == 1): # if spike succeeds (axonal success)
                    activation_delay = int(Ta + np.random.randn()*0.2*Ta)
                    axon_activation[j][t:t+activation_delay] = 0
                    if (random.random() < p[j][t]): # if spike success (synaptic success)
                        if t < t_vec_len:
                            release[j].append(t)
                            p[j][t:] = p[j][t] - Uw*p[j][t]
                            synapse_recover = False

            if (t+1 < t_vec_len) & (synapse_recover):
                p[j][t+1] = p[j][t]+(-p[j][t] + 1)/Tw # synaptic recovery

    return release


if __name__ == "__main__":
    time = 2000 # ms
    hz = 80
    n_axons = 500
    dt = 0.025
    t_vec_len = int(time/dt)
    scs_pulses = [1 if np.mod(x, int(1/hz*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS

    plt.figure(1)
    np.random.seed(1)

    release = get_ia_afferent_release_prob(time, hz, n_axons, p_axonal_success=0.3, Ta=0, Tw=300, delay=0, Uw=0.6)
    total_release = np.array([sum([(0+t in r) for r in release]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    t = np.linspace(0, time, len(total_release))
    plt.figure(1)
    plt.plot(t, total_release/total_release[0], label='Rest')

    release = get_ia_afferent_release_prob(time, hz, n_axons, p_axonal_success=0.3, Ta=20, Tw=300, delay=0, Uw=0.6)
    total_release = np.array([sum([(0+t in r) for r in release]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    t = np.linspace(0, time, len(total_release))
    plt.figure(1)
    plt.plot(t, total_release/total_release[0], label='Max')

    # release = get_ia_afferent_release_prob(time, hz, n_axons, p_axonal_success=0.5, Ta=0, Tw=1000, delay=0)
    # total_release = np.array([sum([(120+t in r) for r in release]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    # plt.figure(1)
    # plt.plot(t, total_release/total_release[0], label='Tw=1000')

    plt.xlabel('Time (ms)')
    plt.ylabel('Release probability')
    plt.legend()

    plt.show()
