import sys
sys.path.append('../ModelingMechanismsSCSinStroke')
from models import MotoneuronNoDendrites
from models.IaAfferentSynapticPotential import get_ia_afferent_synaptic_potential
from neuron import h 
import numpy as np
import matplotlib.pyplot as plt
import random as rnd


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


def run_scs_simulation_rdd(scs_freq, Tw=25, p_axonal_success=1, plot_results=True):
    # model parameters
    num_mns = 5
    num_aff = 0
    num_supra = 60
    simulation_duration = 1000

    # create MN pool
    mn_L = 36+np.random.randn(num_mns)*0.1*36
    np.random.seed(672945) # set the seed so the network is always the same
    drug = True
    mns = [MotoneuronNoDendrites(drug=drug, L=mn_L[imn]) for imn in range(num_mns)]

    # empty arrays to store afferent components
    afferent_inputs = []
    afferent_vecs = []
    afferent_neurons = []
    afferent_neuron_syns = []

    for imn in range(num_mns): 
        afferent_inputs_mn = []
        afferent_vecs_mn = []
        afferent_neurons_mn = []
        afferent_neuron_syns_mn = []

        aff_postsynaptic_potential = get_ia_afferent_synaptic_potential(simulation_duration, 
                                                                        scs_freq=scs_freq, 
                                                                        n_axons=num_aff,
                                                                        Tw=Tw,
                                                                        p_axonal_success=p_axonal_success)
        for n_aff in range(num_aff): 
            # create the afferent point process stimulation to MN soma
            pre = h.NetStim()
            pre.interval = 1000/scs_freq
            pre.noise = 0
            pre.number = 1e999
            pre.start = 3.5 # ms, max latency of spike in synapse
            syn = h.ExpSyn(mns[imn].soma(0.5))
            syn.tau = 2 # ms
            nc = h.NetCon(pre, syn)
            nc.weight[0] = 0

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

    # record SCS spikes
    scs_times = [h.Vector() for i in range(num_aff)]
    scs_detector =  [h.NetCon(afferent_neurons_mn[i], None) for i in range(num_aff)]
    for i in range(num_aff): scs_detector[i].record(scs_times[i])

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

    if plot_results: 
        # plot example MN membranes
        for mem in membrane_record: 
            plt.plot(time_mem, np.array(mem))
            plt.show()



if __name__ == '__main__':
    run_scs_simulation_rdd(80)