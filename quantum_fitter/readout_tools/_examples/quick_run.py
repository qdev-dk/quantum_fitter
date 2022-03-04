import quantum_fitter.readout_tools as rdt
import os

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')
file_path = '/Users/malthenielsen/Documents/GitHub/quantum_fitter/quantum_fitter/readout_tools/examples/example_data/ss_Aalto_190701_rabi_Q4_v2.hdf5'
# Set the instence 
rd = rdt.Readout(file_path, size=100, verbose=1)

# Plot
rd.plot_classifier()

rd.plot_oscillation()
rd.plot_cv_iterations()