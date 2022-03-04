"""import quantum_fitter.readout_tools as rdt
import Labber as lab
import os

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')

file = lab.LogFile(file_path)
data = rdt.reformate(file.getData())

state_0, state_1 = data[0], data[40]
states = [state_0,state_1]

# Set the instence 
rd = rdt.Readout(data=states, size=1000, verbose=1)


# Plot
rd.plot_classifier()

# Use
rd.plot_testing(X=data[1], size=1000)
rd.predict(data[1])
"""
