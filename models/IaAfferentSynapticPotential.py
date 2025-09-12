import random
import numpy as np
import matplotlib.pyplot as plt

n_vesicle = 2

def get_ia_afferent_release_prob(sim_dur, 
                                scs_freq, 
                                n_axons, 
                                Tb=100, 
                                Ta=20,
                                Tf=300, 
                                Ub=0.7, # probability of synaptic release
                                Ua=0.7, # OG: 0.0025, probability of axonal refractory period
                                Uf=0.99, # facilitation increment per spike
                                dt=0.025,
                                F0=0.5,
                                seed=672944, 
                                plot=False): 
    """
    Function for modeling axonal and synaptic failure in response to stimulation. 

    SCS spikes constitute a "nascent spike", which may or may not travel to the MN itself, 
    either because of axonal or synaptic failure.

    The following function predicts the success or failure of spikes after axonal and synaptic failure.
    """
    np.random.seed(seed)

    # adjust time-based parameters to dt 
    Ta = Ta/dt # time constant for recovery of axon # OG: 27
    Tb = Tb/dt # dt steps, mean of exponentially-distributed random variable for wait time before recovery, OG: 850
    Tf = Tf/dt # dt steps, time constant for facilitation

    # set up variables 
    t_vec_len = int(sim_dur/dt)
    p = np.ones([n_axons, t_vec_len]) # probability of vesicle release
    n_all = n_vesicle*np.ones([n_axons, t_vec_len]) # docked vesicles in each synapse
    x = np.ones([n_axons, t_vec_len]) # probability of action potential success in each axon, at each time t
    F = np.ones([n_axons, t_vec_len])*F0 # facilitation factors
    release = [[] for _ in range(n_axons)] # time of successful spikes
    scs_pulses = [1 if np.mod(x, int(1/scs_freq*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS

    # run simulation of axonal and synaptic failure, predict spike train
    for t in range(t_vec_len - 1): 
        for j in range(n_axons): 
            x[j][t+1] = x[j][t] + (-x[j][t] + 1)/Ta # axonal recovery
            F[j][t+1] = F[j][t] * (1 - 1/Tf) # facilitation decay
            if (scs_pulses[t] == 1) & (random.random() <= 1): # if SCS pulse + additional randomness   
                if (random.random() <= x[j][t]): # if axonal success
                    p[j][t] = 1-(1-Ub)**n_all[j][t] 
                    p[j][t] *= F[j][t] # facilitation impact on p(synapse)
                    
                    if random.random() <= p[j][t]: # if synaptic success 
                        redock_time = np.random.exponential(Tb)
                        redock_time = int(np.round(np.min([t_vec_len-t, redock_time])))
                        n_all[j][t:] = np.max([0, n_all[j][t]-1])
                        n_all[j][t+redock_time:] += 1 
                        release[j].append(t*dt)
                    
                    F[j][t+1] = F[j][t] + (1 - F[j][t]) * Uf # facilitation increase with successful spike

                x[j][t+1] = x[j][t] - Ua*x[j][t] # axonal refractory period
            
    if plot: 
        plt.figure(2)
        inds = [s == 1 for s in scs_pulses]
        x_plot = [a[inds] for a in x]
        plt.plot(np.mean(x_plot, axis=0), label=f'Ta={Ta*dt}')
        plt.title('Probability of axonal success')

        plt.figure(3)
        p_plot = [s[inds] for s in p]
        plt.plot(np.mean(p_plot, axis=0), label=f'Ta={Ta*dt}')
        plt.title('Probability of synaptic success')

        plt.figure(4)
        f_plot = [f[inds] for f in F]
        plt.plot(np.mean(f_plot, axis=0), label=f'Ta={Ta*dt}')
        plt.title('Synaptic facilitation')
    return release


if __name__ == "__main__":
    time = 12.5*100 # ms
    hz = 80
    n_axons = 500
    dt = 0.025
    t_vec_len = int(time/dt)
    scs_pulses = [1 if np.mod(x, int(1/hz*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS

    start_ind = 0

    # max
    Ta = 5
    release_control_ux = get_ia_afferent_release_prob(time, hz, n_axons, Ta=Ta, plot=True)
    total_release = np.array([sum([(t*dt in r) for r in release_control_ux]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    t = np.linspace(0, time, len(total_release))
    plt.figure(1)
    plt.plot(t[start_ind:], total_release[start_ind:]/total_release[start_ind], label=f'Ta={Ta}')

    plt.figure(5)
    max_ind = np.argmax(total_release)
    plt.plot(t[:len(t)-max_ind], total_release[max_ind:]/np.max(total_release), label=f'Ta={Ta}')

    # rest
    Ta = 50
    release_high_ux = get_ia_afferent_release_prob(time, hz, n_axons, Ta=Ta, plot=True)
    total_release = np.array([sum([(t*dt in r) for r in release_high_ux]) for t in range(t_vec_len) if scs_pulses[t] == 1])
    plt.figure(1)
    plt.plot(t[start_ind:], total_release[start_ind:]/total_release[start_ind], label='Ta={Ta}')
    plt.legend()

    plt.figure(5)    
    max_ind = np.argmax(total_release)
    plt.plot(t[:len(t)-max_ind], total_release[max_ind:]/np.max(total_release), label=f'Ta={Ta}')
    plt.legend()

    plt.figure(2)
    plt.legend()

    plt.figure(3)
    plt.legend()


    plt.show()



    




# Works?? w/ rest = 5, max = 50; need less depression still
#     n_vesicle = 2

# def get_ia_afferent_release_prob(sim_dur, 
#                                 scs_freq, 
#                                 n_axons, 
#                                 Tb=100, 
#                                 Ta=20,
#                                 Tf=300, 
#                                 Ub=0.7, # probability of synaptic release
#                                 Ua=0.7, # OG: 0.0025, probability of axonal refractory period
#                                 Uf=0.99, # facilitation increment per spike
#                                 dt=0.025, 
#                                 F0=0.5,
#                                 seed=672945, 
#                                 plot=False): 