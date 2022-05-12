import quantum_fitter.readout_tools as rdt
import os
import numpy as np
import Labber as lab
import matplotlib.pyplot as plt

# Set up path
dir = os.path.dirname(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir))
#file_path = os.path.join(dir, '_examples/example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')
file_path2 = os.path.join(dir, '_examples/example_data/ss_Aalto_190701_rabi_Q4_v2.hdf5')
file_path1 = '/Users/malthenielsen/Desktop/amplitude rabi ss.hdf5'


# Set the instence 
rd = rdt.Readout(file_path1, channelName='Multi-Qubit Pulse Generator - Single-shot, QB1', size=1000, verbose=1, state_entries=[0,10])
#rd.plot_classifier()
#rd.plot_temp()

rd2 = rdt.Readout(file_path2, size=100, verbose=1)
ax2 = rd2.plot_oscillation()

ax1 = rd.plot_oscillation()



file = lab.LogFile(file_path)
x = file.getData('Multi-Qubit Pulse Generator - Single-shot, QB1')


XX,YY = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]))
table = np.vstack((x.ravel(),XX.ravel(),YY.ravel())).T


fig = plt.figure(figsize=(8,6))
plt.hexbin(table[:,0].real,table[:,1].real)
plt.title("Plot 2D array")
plt.colorbar()
plt.show()
