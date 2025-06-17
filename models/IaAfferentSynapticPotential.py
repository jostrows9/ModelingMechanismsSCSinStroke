import random
import numpy as np
import matplotlib.pyplot as plt

### STATIC PARAMETERS
Ux = 2.5e-3 # 0 < Ux < 1, probability that each spike impacts probability of spiking
Ul = 1.5e-2 # 0 < Ul < 1, probability that each spike impacts latency
Ux_pop = 2e-3 # 0 < Ux_pop < 1, probability that spikes in one axon impact the surrounding population (TODO: remove this?)
Cm = 1 # uF/cm^2
J = 0.085*Cm # synaptic weight, tuned to single EPSP of ~200uV
Uw = 0.6 # Uw < 1, probability of vesicle release, OG: 0.06
n_vesicle = 5 # number of vesicle docking sites

def calculate_alpha(t, T1, T2): 
    return (J/(T2-T1))*np.heaviside(t, t)*(np.exp(-t/T2)-np.exp(-t/T1)) # post-synaptic conductance waveform

def get_ia_afferent_synaptic_potential(sim_dur, scs_freq, n_axons, Tw=25, p_axonal_success=1, dt=0.025): 
    """
    Function for modeling axonal and synaptic failure in response to stimulation. 

    Somatic spikes: innate Ia-firing (TODO: figure out if we remove this, or use this as proprioceptive info that must be maintained)
    Spinal cord stimulation: electrical stimulation to Ia-afferents.

    Somatic and SCS spikes constitute a "nascent spike", which may or may not travel to the MN itself, 
    either because of axonal or synaptic failure.

    The following function predicts the postsynaptic conductance after axonal and synaptic failure.
    """
    
    # adjust time-based parameters to dt 
    L_max = 3.5/dt # dt steps, maximum latency
    L_min = 2.8/dt # minimum latency 
    T_x = 27/dt # time constant for recovery 
    T_L = 27/dt # time constant for recovery (latency)
    Tw = Tw/dt # dt steps, mean of exponentially-distributed random variable for wait time before recovery, OG: 850
    T1 = 1/dt # dt steps
    T2 = 4/dt # dt steps

    # set up variables 
    t_vec_len = int(sim_dur/dt)
    g = np.zeros([n_axons, t_vec_len]) # postsynaptic conductance 
    n_all = n_vesicle*np.ones([n_axons, t_vec_len]) # docked vesicles in each synapse, 0 < n_j < 5
    p = np.ones([n_axons, t_vec_len]) # probability of vesicle release
    x = np.ones([n_axons, t_vec_len])*p_axonal_success # probability of action potential success in each axon, at each time t
    L = L_min*np.ones([n_axons, t_vec_len]) # latency of spike at each time t 
    release = [[] for _ in range(n_axons)] # time of release of vesicles
    scs_pulses = [1 if np.mod(x, int(1/scs_freq*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS
    somatic_pulses = [[-1] for _ in range(n_axons)] # TODO: for now, no somatic pulses, needs to be different for each axon

    # run simulation of axonal and synaptic failure, predict postsynaptic potential
    for t in range(t_vec_len): 
        for j in range(n_axons): 
            if (scs_pulses[t] == 1) | (t in somatic_pulses[j]):      
                if (random.random() < x[j][t]): # if spike succeeds
                    t_spike = t + int(np.round(L[j][t])) # spike time
                    if t_spike < t_vec_len:
                        p[j][t_spike] = 1-(1-Uw)**n_all[j][t]
                        redock_time = np.random.exponential(Tw)
                        redock_time = int(np.round(np.min([t_vec_len-t_spike, redock_time])))
                        n_all[j][t_spike:] = np.max([0, n_all[j][t_spike]-1])
                        n_all[j][t_spike+redock_time:] += 1 
                        if random.random() < p[j][t_spike]: 
                            release[j].append(t_spike)

                x[j][t:] = x[j][t] - Ux*x[j][t]
                L[j][t:] = L[j][t] + Ul*(L_max - L[j][t])
            
            elif t+1 < t_vec_len:
                x[j][t+1] = x[j][t]+(-x[j][t] + 1)/T_x # axonal recovery
                L[j][t+1] = L[j][t]+(-L[j][t] + L_min)/T_L # latency recovery

            g[j][t] = np.sum([calculate_alpha(t-release_t, T1, T2) for release_t in release[j]]) 
    
    return g 


if __name__ == "__main__":
    time = 2000 # ms
    hz = 80
    n_axons = 1

    g = get_ia_afferent_synaptic_potential(time, hz, n_axons)#, dt=1)
    import pdb; pdb.set_trace()
    plt.plot(g[0])
    plt.show()
