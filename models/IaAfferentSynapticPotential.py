import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

### STATIC PARAMETERS
Cm = 2 # uF/cm^2
J = 0.085*Cm # synaptic weight, tuned to single EPSP of ~200uV
Uw = 0.6 # Uw < 1, probability of vesicle release, OG: 0.06
n_vesicle = 5 # number of vesicle docking sites

def calculate_alpha(t, T1, T2, syn_w): 
    return (syn_w/(T2-T1))*np.heaviside(t, t)*(np.exp(-t/T2)-np.exp(-t/T1)) # post-synaptic conductance waveform

def get_ia_afferent_synaptic_potential(sim_dur, 
                                       scs_freq, 
                                       n_axons, 
                                       Tw=25, 
                                       p_axonal_success=1, 
                                       Ux=2.5e-3,
                                       dt=0.025): 
    # check if previous-run exists 
    file_path = f"data/synaptic_potentials/synaptic_potential_SCS_freq_{scs_freq}_n_axons_{n_axons}_tw_{Tw}_p_success_{p_axonal_success}_dt_{dt}_simulation_duration_{sim_dur}_Ux_{Ux}.pickle"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f: 
            data = pickle.load(f)
            print("Synaptic potentials file already exists. Returning loaded data.")
            return data['conductance']
    
    # set up variables related to conductance
    t_vec_len = int(sim_dur/dt)
    g = np.zeros([n_axons, t_vec_len]) # postsynaptic conductance 
    T1 = 1/dt # dt steps
    T2 = 4/dt # dt steps
    
    # random synaptic weight per axon
    synaptic_shape = 1.2
    syn_ws = np.random.gamma(synaptic_shape, scale=J/synaptic_shape, size=n_axons)

    release_prob = get_ia_afferent_release_prob(sim_dur, 
                                                scs_freq=scs_freq, 
                                                n_axons=n_axons, 
                                                Tw=Tw, 
                                                p_axonal_success=p_axonal_success, 
                                                Ux=Ux, 
                                                dt=dt)
    
    for t in range(t_vec_len): 
        for j in range(n_axons): 
            g[j][t] = np.sum([calculate_alpha(t-release_t, T1, T2, syn_ws[j]) for release_t in release_prob[j]]) 

    # save data
    data = {}
    data['conductance'] = g
    data['release'] = release_prob
    with open(file_path, 'wb') as f: 
        pickle.dump(data, f)

    return g


def get_ia_afferent_release_prob(sim_dur, 
                                       scs_freq, 
                                       n_axons, 
                                       Tw=25, 
                                       p_axonal_success=1, 
                                       Ux=2.5e-3,
                                       dt=0.025): 
    """
    Function for modeling axonal and synaptic failure in response to stimulation. 

    Somatic spikes: innate Ia-firing (TODO: figure out if we remove this, or use this as proprioceptive info that must be maintained)
    Spinal cord stimulation: electrical stimulation to Ia-afferents.

    Somatic and SCS spikes constitute a "nascent spike", which may or may not travel to the MN itself, 
    either because of axonal or synaptic failure.

    The following function predicts the postsynaptic conductance after axonal and synaptic failure.
    """

    # adjust time-based parameters to dt 
    L = 3/dt # spike latency 
    T_x = 100/dt # time constant for recovery # OG: 27
    Tw = Tw/dt#Tw/dt # dt steps, mean of exponentially-distributed random variable for wait time before recovery, OG: 850

    # set up variables 
    t_vec_len = int(sim_dur/dt)
    n_all = np.random.poisson(lam=n_vesicle, size=[n_axons, t_vec_len]) # docked vesicles in each synapse, 0 < n_j < 5
    p = np.ones([n_axons, t_vec_len]) # probability of vesicle release
    x = np.ones([n_axons, t_vec_len])*p_axonal_success # probability of action potential success in each axon, at each time t
    release = [[] for _ in range(n_axons)] # time of release of vesicles
    scs_pulses = [1 if np.mod(x, int(1/scs_freq*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS

    # run simulation of axonal and synaptic failure, predict postsynaptic potential
    for t in range(t_vec_len): 
        for j in range(n_axons): 
            if (scs_pulses[t] == 1):      
                if (random.random() < x[j][t]): # if spike succeeds
                    t_spike = t + int(L) # spike time
                    if t_spike < t_vec_len:
                        p[j][t_spike] = 1-(1-Uw)**n_all[j][t]
                        redock_time = np.random.exponential(Tw)
                        redock_time = int(np.round(np.min([t_vec_len-t_spike, redock_time])))
                        n_all[j][t_spike:] = np.max([0, n_all[j][t_spike]-1])
                        n_all[j][t_spike+redock_time:] += 1 
                        if random.random() < p[j][t_spike]: 
                            release[j].append(t_spike)

                x[j][t:] = x[j][t] - Ux*x[j][t]
            
            elif t+1 < t_vec_len:
                x[j][t+1] = x[j][t]+(-x[j][t] + 1)/T_x # axonal recovery

    plt.figure(1)
    plt.plot(x[0], label=f"{p_axonal_success} % starting axonal success, {Ux} % decrease axonal success")
    return release


if __name__ == "__main__":
    time = 200 # ms
    hz = 80
    n_axons = 500
    dt = 0.025
    t_vec_len = int(time/dt)
    scs_pulses = [1 if np.mod(x, int(1/hz*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS

    # max
    release_control_ux = get_ia_afferent_release_prob(time, hz, n_axons, p_axonal_success=0.9)
    total_release = np.array([sum([(120+t in r) for r in release_control_ux]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    t = np.linspace(0, time, len(total_release))
    plt.figure(2)
    plt.plot(t, total_release/total_release[0], label='Max')

    # rest
    release_high_ux = get_ia_afferent_release_prob(time, hz, n_axons, p_axonal_success=0.5)
    total_release = np.array([sum([(120+t in r) for r in release_high_ux]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    
    plt.figure(2)
    plt.plot(t, total_release/total_release[0], label='Rest')

    plt.legend()

    plt.figure(1)
    plt.legend()
    plt.show()
