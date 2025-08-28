from itertools import product
import pickle
import sys
sys.path.append('../ModelingMechanismsSCSinStroke')
from models import MotoneuronNoDendrites
from models.SimpleSynapticPotentialModel import get_ia_afferent_synaptic_potential
from neuron import h 
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import os
from multiprocessing import Pool

def plot_simulation_results(data): 

    plt.subplots(4, 1, sharex=True)

    # plot example MN membranes
    plt.subplot(4,1,1)
    for mem in data['membrane_potentials']: 
        plt.plot(data['simulation_time_vector'], np.array(mem))
    plt.ylabel('Membrane\n Potential (mV)')

    # plot example Ia-afferent postsynaptic conductance
    plt.subplot(4,1,2)
    n_aff_to_plot = data['num_scs_total']

    for aff_ind in range(n_aff_to_plot): 
        ex_aff_potential = data['g_scs'][aff_ind]
        plt.plot(data['simulation_time_vector'][:len(ex_aff_potential)], np.array(ex_aff_potential))
    plt.ylabel('Ia-Afferent\n Conductance')

    # plot SCS pulses
    plt.subplot(4,1,3)
    for i in range(data['num_scs_total']): plt.plot(data["scs_pulse_times"][i], i*np.ones(len(data["scs_pulse_times"][i])), ".", c='grey', markersize=1)
    plt.ylabel('Afferent Input')

    # plot supraspinal firing
    plt.subplot(4,1,4)
    for i in range(data['num_supraspinal']): plt.plot(data["supraspinal_spike_times"][i], i*np.ones(len(data["supraspinal_spike_times"][i])), ".", c='grey', markersize=1)
    plt.ylabel('Supraspinal\n Input')
    plt.xlabel('Time (ms)')

    plt.tight_layout()

    plt.subplots(3, 1, sharex=True)

    # plot MN firing
    plt.subplot(3,1,1)
    for i in range(data['num_mn']): plt.plot(data["mn_spikes"][i], i*np.ones(len(data["mn_spikes"][i])), ".", c='grey', markersize=2)
    plt.ylabel('Motoneuron\n Firing')

    # plot EMG signal
    plt.subplot(3,1,2)
    plt.plot([t for t in range(len(data["emg"]))], data["emg"])
    plt.ylabel('EMG Signal\n (mV)')

    # plot normalized P2P amplitude
    plt.subplot(3,1,3)
    if (len(data["p2p_amp"]) > 0): 
        if (data["p2p_amp"][0] != 0):
            evoked_response_time = [int(pulseTime)+15 for pulseTime in data["scs_pulse_times"][0]]
            plt.plot(evoked_response_time[:len(data["p2p_amp"])], data["p2p_amp"]/data["p2p_amp"][0])
            plt.ylabel('Normalized P2P\n Amplitude (%)')
            plt.xlabel('Time (ms)')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def estimateP2PAmp(emg, scsParadigm, delay_ms=10, dt=1): 
    p2p = []
    if len(scsParadigm) > 0:
        scsPulses = [int(pulseTime) for pulseTime in scsParadigm[0]]
        for pulseInd in scsPulses: 
            start_response_ind = pulseInd + int(delay_ms/dt)
            end_response_ind = pulseInd + int(delay_ms+10/dt)
            if end_response_ind < len(emg):
                p2pPulse = np.max(emg[start_response_ind:end_response_ind])-np.min(emg[start_response_ind:end_response_ind])
                if p2pPulse > 0: 
                    p2p.append(p2pPulse)
                else: 
                    p2p.append(0)
    return p2p


def estimateEMG(firings, delay_ms=10):
    """
    Estimate the EMG activity given the cell firings. 

    Delay is the delay between the motoneuron action potential and the motor unit action potential (MUAP).
    """
    nCells = len(firings)
    nSamples = firings.shape[1]

    # MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-1 
    meanLenMUAP = 8
    stdLenMUAP = 1

    nS = [int(meanLenMUAP+rnd.gauss(0,stdLenMUAP)) for _ in range(nCells)]
    amp = [abs(1+rnd.gauss(0,0.2)) for _ in range(nCells)]
    EMG = np.zeros(nSamples + max(nS)+delay_ms)
    firingInd = []
    # create MUAP shape
    for i in range(nCells):
        n40perc = int(nS[i]*0.4)
        n60perc = nS[i]-n40perc
        amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) * np.concatenate((np.ones(n40perc),1/np.linspace(1,3,n60perc)))
        logBase = 1.05
        freqMod = np.log(np.linspace(1,logBase**(4*np.pi),nS[i]))/np.log(logBase)
        EMG_unit = amp[i]*amplitudeMod*np.sin(freqMod)
        for j in range(nSamples):
            if firings[i,j]==1:
                firingInd.append(j)
                EMG[j+delay_ms:j+delay_ms+nS[i]]=EMG[j+delay_ms:j+delay_ms+nS[i]]+EMG_unit
    
    EMG = EMG[:nSamples]
    
    return EMG


def run_scs_simulation_rdd(scs_freq, 
                           num_supra=20, 
                           supra_freq=60, 
                           num_aff=30,
                           Tw=80, 
                           Ta=20,
                           p_axonal_success=1, 
                           Uw=0.6,
                           simulation_duration=200,
                           plot_results=True, 
                           save_data_folder='',
                           seed=672945):
    
    # model parameters
    num_mns = 100
    synaptic_weight = 0.000148
    synaptic_shape = 1.2

    # create MN pool
    np.random.seed(seed) # set the seed so the network is always the same
    mn_L = 36+np.random.randn(num_mns)*0.1*36
    
    drug = True
    mns = [MotoneuronNoDendrites(drug=drug, L=mn_L[imn]) for imn in range(num_mns)]

    # build and store afferent components
    afferent_inputs = []
    afferent_vecs = []
    afferent_neurons = []
    afferent_neuron_syns = []

    for imn in range(num_mns): 
        afferent_inputs_mn = []
        afferent_vecs_mn = []
        afferent_neurons_mn = []
        afferent_neuron_syns_mn = []

        
        aff_postsynaptic_potential = get_ia_afferent_synaptic_potential(
                                                    sim_dur=simulation_duration,
                                                    scs_freq=scs_freq,
                                                    n_axons=num_aff,
                                                    Tw=Tw,
                                                    Uw=Uw,
                                                    p_axonal_success=p_axonal_success,
                                                    seed=seed,
                                                )
        
        for n_aff in range(num_aff): 
            # create the afferent point process stimulation to MN soma
            pre = h.NetStim()
            pre.interval = 1000/scs_freq
            pre.noise = 0
            pre.number = 1e999
            pre.start = 50 # ms, delay
            syn = h.ExpSyn(mns[imn].soma(0.5))
            syn.tau = 2 # ms
            nc = h.NetCon(pre, syn)
            nc.weight[0] = synaptic_weight

            # build and play change in conductance
            g_ramp = aff_postsynaptic_potential[n_aff]
            g_vec = h.Vector(g_ramp)
            g_vec.play(nc._ref_weight[0], 0.025)
            
            # store afferent components per afferent
            afferent_inputs_mn.append(nc)
            afferent_vecs_mn.append(g_vec)
            afferent_neurons_mn.append(pre)
            afferent_neuron_syns_mn.append(syn)

        # store afferent components all
        afferent_inputs.append(afferent_inputs_mn)
        afferent_vecs.append(afferent_vecs_mn)
        afferent_neurons.append(afferent_neurons_mn)
        afferent_neuron_syns.append(afferent_neuron_syns_mn)

    # build and store supraspinal inputs
    
    supra_W = np.random.gamma(synaptic_shape, scale=synaptic_weight/synaptic_shape, size=[num_supra, num_mns])

    supra_neurons = []
    for supra_ind in range(num_supra):
        # create the supra point process stimulation to MN soma
        pre = h.NetStim()
        pre.interval = 1000/supra_freq
        pre.noise = 1
        pre.number = 1e999
        pre.start = 50 # ms
        supra_neurons.append(pre)

    supra_inputs = []
    supra_neurons_syns = []
    for mn_ind in range(num_mns):
        supra_inputs_mn = []
        supra_neurons_syns_mn = []
        for supra_ind in range(num_supra):
            syn = h.ExpSyn(mns[mn_ind].soma(0.5))
            syn.tau = 2 # ms
            nc = h.NetCon(supra_neurons[supra_ind], syn)
            nc.weight[0] = supra_W[supra_ind][mn_ind]

            supra_inputs_mn.append(nc)
            supra_neurons_syns_mn.append(syn)
        
        supra_inputs.append(supra_inputs_mn)
        supra_neurons_syns.append(supra_neurons_syns_mn)

    # record SCS spikes
    scs_times = [h.Vector() for i in range(num_aff)]
    scs_detector =  [h.NetCon(afferent_neurons_mn[i], None) for i in range(num_aff)]
    for i in range(num_aff): scs_detector[i].record(scs_times[i])

    # record supraspinal spikes
    supra_times = [h.Vector() for i in range(num_supra)]
    supra_detector =  [h.NetCon(supra_neurons[i], None) for i in range(num_supra)]
    for i in range(num_supra): supra_detector[i].record(supra_times[i])

    # record MN spikes
    mn_times = [h.Vector() for i in range(num_mns)]
    mn_detector = []
    for i in range(num_mns):
        sp_detector = h.NetCon(mns[i].soma(0.5)._ref_v, None, sec=mns[i].soma)
        sp_detector.threshold = -5
        mn_detector.append(sp_detector)
        mn_detector[i].record(mn_times[i])

    if plot_results: 
        # record example MN membranes
        record_mn_num = np.min((num_mns, 5))
        membrane_record = [h.Vector().record(mns[i].soma(0.5)._ref_v) for i in range(record_mn_num)]
        time_mem = h.Vector().record(h._ref_t)

    # run simulation
    h.load_file('stdrun.hoc')
    h.finitialize()
    h.tstop = simulation_duration
    h.run()

    supra_times = [np.array(supra_times[i]) if len(supra_times[i]) > 0 else [] for i in range(num_supra)]
    scs_times = [np.array(scs_times[i]) if len(scs_times[i]) > 0 else [] for i in range(num_aff)]
    mn_times = [np.array(mn_times[i]) if len(mn_times[i]) > 0 else [] for i in range(num_mns)]
    
    membrane_potentials = []
    if plot_results: 
        membrane_potentials = [np.array(membrane_record[i]) if len(membrane_record[i]) > 0 else [] for i in range(record_mn_num)]
        time_mem = np.array(time_mem)
    else: 
        time_mem = []

    firings_int = [[int(spike) for spike in mn_firings] for mn_firings in mn_times]
    firings_mat = np.array([[1 if i in mn_firings else 0 for i in range(0, simulation_duration)] for mn_firings in firings_int])

    emg_signal = estimateEMG(firings_mat)
    p2p_amp = estimateP2PAmp(emg_signal, scs_times)
    
    data={}
    data["mn_spikes"] = mn_times
    data["supraspinal_spike_times"] = supra_times
    data["scs_pulse_times"] = scs_times
    data["scs_frequency"] = scs_freq
    data["num_scs_total"] = num_aff
    data["g_scs"] = aff_postsynaptic_potential
    data["supraspinal_rate"] = supra_freq
    data["num_supraspinal"] = num_supra
    data["simulation_duration"] = simulation_duration
    data["num_mn"] = num_mns
    data["synaptic_weight_supra"] = synaptic_weight
    data["mn_L"] = mn_L
    data["simulation_time_vector"] = time_mem
    data["membrane_potentials"] = membrane_potentials
    data["p2p_amp"] = p2p_amp
    data["emg"] = emg_signal
    data_filename = f"mnNum_{num_mns}_supraspinalNum_{num_supra}_supraspinalFR_{supra_freq}_SCSFreq_{scs_freq}_SCSTotal_{num_aff}_SynW_{synaptic_weight}_tw_{Tw}_uw_{Uw}_ta_{Ta}_p_success_{p_axonal_success}_seed_{seed}.pickle"
    
    if save_data_folder != '': 
        ensure_dir(save_data_folder)

        f=open(save_data_folder+data_filename,"wb")
        pickle.dump(data,f)
        f.close()

    if plot_results: 
        plot_simulation_results(data)
        
    return data_filename


if __name__ == '__main__':
    scs_freq = 80

    #run_scs_simulation_rdd(scs_freq, num_supra=0, num_aff=30, plot_results=True, p_axonal_success=0.3, simulation_duration=200, Tw=40, Uw=0.6, seed=1)
    #run_scs_simulation_rdd(scs_freq, num_supra=15, num_aff=30, plot_results=True, p_axonal_success=0.6, simulation_duration=200, Tw=40, Uw=0.6, seed=1)
    
    #plt.show()

    for i in range(20):
        # longer simulation runs
        run_scs_simulation_rdd(scs_freq, num_supra=0, num_aff=30, plot_results=False, p_axonal_success=0.3, Uw=0.6, Tw=40, simulation_duration=2000, seed=i, save_data_folder=f'data/old_runs/2000ms_runs/rest_data_0.3_tw_40/') # rest
        run_scs_simulation_rdd(scs_freq, num_supra=15, num_aff=30, plot_results=False, p_axonal_success=0.6, Uw=0.6, Tw=40, simulation_duration=2000, seed=i, save_data_folder='data/old_runs/2000ms_runs/25%_data_0.6_tw_40/') # max, just reuptake
        #run_scs_simulation_rdd(scs_freq, num_supra=15, num_aff=30, plot_results=False, p_axonal_success=0.3, Uw=0, simulation_duration=2000, seed=i, save_data_folder='data/no_rdd_model/25%_data_0.3_uw_0/') # max, no pad

        # # test tw values
        # tw_opts = [20, 500]
        # for tw in tw_opts: 
        #     run_scs_simulation_rdd(scs_freq, num_supra=0, num_aff=30, plot_results=False, p_axonal_success=0.3, Ta=0, Tw=tw, seed=i, save_data_folder=f'data/2000ms_runs/rest_data_0.3_tw_{tw}/') # rest
        #     run_scs_simulation_rdd(scs_freq, num_supra=15, num_aff=30, plot_results=False, p_axonal_success=0.6, Ta=0, Tw=tw, seed=i, save_data_folder=f'data/2000ms_runs/25%_data_0.6_tw_{tw}/') # max
