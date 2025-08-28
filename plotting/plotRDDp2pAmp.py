import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def double_exponential(x, A1, k1, A2, k2, C):
    return A1 * np.exp(-x/k1) + A2 * np.exp(-x/k2) + C

title = 'Reuptake Model Alone'
path = 'data/old_runs/2000ms_runs'
#path = 'data/old_runs/my_model/200ms_runs'
#path = 'data/no_rdd_model'
dataFolders = {'No PAD or Reuptake Model': {'25% Effort': '25%_data_0.5_tw_80', 
                                'Rest': 'rest_data_0.3_tw_20'}, 
                'PAD & Reuptake Model': {'25% Effort': '25%_data_0.5_tw_20', 
                    'Rest': 'rest_data_0.3_tw_80'}, 
               'PAD Model Alone': {'25% Effort': '25%_data_0.5_tw_80', 
                    'Rest': 'rest_data_0.3_tw_40'}, 
               'Reuptake Model Alone': {'25% Effort': '25%_data_0.3_tw_20', 
                    'Rest': 'rest_data_0.3_tw_80'}
}
conditions = ["25% Effort", 'Rest']
colors = ["#b01c3f", "#23b0cc"]
numCon = len(conditions)
hz = 80
delay = 10
emgLength = 12

p2pNormData = {}
emgAmps = {}

for conInd in range(numCon): 
    folder = f"{path}/{dataFolders[title][conditions[conInd]]}"
    p2pNormData[conditions[conInd]] = []
    emgAmps[conditions[conInd]] = []
    for filename in os.listdir(folder):
        if filename.endswith('.pickle'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'rb') as f: 
                data = pickle.load(f)
                p2pNormData[conditions[conInd]].append(data['p2p_amp']/data['p2p_amp'][0])

                scsTimes = data['scs_pulse_times'][0]
                emgData = data['emg']
                for t in scsTimes[3:]: 
                    responseStart = int(t+delay)
                    responseEnd = responseStart+emgLength
                    if responseEnd < len(emgData):
                        resp = emgData[responseStart:responseEnd]
                        emgAmps[conditions[conInd]].append(np.max(resp)-np.min(resp))

for conInd in range(numCon): 
    p2pDataAvg = np.average(p2pNormData[conditions[conInd]], axis=0)
    p2pDataStd = np.std(p2pNormData[conditions[conInd]], axis=0)

    x_data = np.array([int(x/hz*1000) for x in range(0,len(p2pDataAvg))])
    initial_guess = [0.5, 15, 0.5, 1000, 0.5]
    params, covariance = curve_fit(double_exponential, x_data, p2pDataAvg, p0=initial_guess)
    A1_fit, k1_fit, A2_fit, k2_fit, C_fit = params
    y_fit = double_exponential(x_data, A1_fit, k1_fit, A2_fit, k2_fit, C_fit)
    print(f"Params: {params}")
    
    plt.figure(1)
    plt.plot(x_data, p2pDataAvg, c=colors[conInd], label=conditions[conInd], marker='o')
    plt.fill_between(x_data, p2pDataAvg-p2pDataStd, p2pDataAvg+p2pDataStd, color=colors[conInd], alpha=0.05)
    plt.plot(x_data, y_fit, color=colors[conInd], linestyle='--', label=f'Fitted Double Exponential: {conditions[conInd]}')
    plt.ylabel('Normalized P2P Amplitude (% of max)')
    plt.xlabel('Time (ms)')
    plt.title(f'RDD in P2P Amplitude: {title}')
    plt.legend()
    
    plt.figure(2)
    avg_emg_amp = np.mean(emgAmps[conditions[conInd]])
    
    plt.subplot(2, 2, 1)
    plt.scatter(k1_fit, avg_emg_amp, marker='x', color=colors[conInd])
    plt.xlabel('T1 decay rate (ms)')
    plt.ylabel('C (uV)')

    plt.subplot(2, 2, 2)
    plt.scatter(k2_fit, avg_emg_amp, marker='x', color=colors[conInd])
    plt.xlabel('T2 decay rate (ms)')
    plt.ylabel('C (uV)')

    plt.subplot(2, 2, 3)
    tau_effective = (A1_fit*k1_fit + A2_fit*k2_fit)/(A1_fit+A2_fit)
    plt.scatter(tau_effective, avg_emg_amp, marker='x', color=colors[conInd])
    plt.xlabel('T-effective decay rate (ms)')
    plt.ylabel('C (uV)')

    plt.subplot(2, 2, 4)
    plt.scatter(A1_fit/A2_fit, avg_emg_amp, marker='x', color=colors[conInd])
    plt.xlabel('A1/A2')
    plt.ylabel('C (uV)')
    
    plt.suptitle(f'Exponential Fit Metrics: {title}')
    
    plt.tight_layout()



plt.figure(1)
plt.savefig(f'plotting/plots/{title}_p2p_decay.png', dpi=300)

plt.figure(2)
plt.savefig(f'plotting/plots/{title}_exponential_fitted_metrics.png', dpi=300)

plt.show()
