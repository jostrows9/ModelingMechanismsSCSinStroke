import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataFolders = ['data/25%_data_0.5_tw_20', 'data/rest_data_0.3_tw_80']
conditions = ["25% Effort", 'Rest']
colors = ["#b01c3f", "#23b0cc"]
numCon = len(conditions)
hz = 80
delay = 10
emgLength = 12

emgResponses = {}

for conInd in range(numCon): 
    folder = dataFolders[conInd]
    emgResponses[conditions[conInd]] = []
    for filename in os.listdir(folder):
        if filename.endswith('.pickle'):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'rb') as f: 
                data = pickle.load(f)
                scsTimes = data['scs_pulse_times'][0]
                emgData = data['emg']
                for t in scsTimes: 
                    responseStart = int(t+delay)
                    responseEnd = responseStart+emgLength
                    if responseEnd < len(emgData):
                        resp = emgData[responseStart:responseEnd]
                        if np.sum(resp) > 0: 
                            plt.figure(1)
                            plt.plot(resp, c=colors[conInd], alpha=0.1)
                        emgResponses[conditions[conInd]].append(resp)
    plt.figure(2)
    emgAvgResp = np.mean(emgResponses[conditions[conInd]], axis=0)
    plt.plot(emgAvgResp, c=colors[conInd], label=conditions[conInd])
    plt.xlabel('Time (ms)')
    plt.ylabel('Estimate EMG Amplitude (V)')
    plt.title('Average Esimtated EMG Trace: \n80 Hz Stimulation')




plt.legend()
plt.savefig('plotting/plots/estimatedEMGTrace.png')
plt.show()
