import quantum_fitter.readout_tools as rdt
import os

import matplotlib.pyplot as plt

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')


# Set the instence 
rd = rdt.Readout(file_path, size=200, verbose=1)
rd.set_states(state_entries=[2,39]) # makeing 'fake' 3 state data

# Plot
#rd.plot_oscillation(size=100, mode = 'expectation', state=0)

rd.plot_oscillation(size=1000, mode='expectation', state='all')
