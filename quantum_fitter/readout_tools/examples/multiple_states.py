"""import quantum_fitter.readout_tools as rdt
import os


# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')


# Set the instence 
rd = rdt.Readout(file_path, size=2000, verbose=1)
rd.set_states(state_entries=[2,39, 20], offset=-0.00025) # makeing 'fake' 3 state data

# Plot
rd.plot_classifier()
rd.plot_testing()
rd.plot_ROC()
rd.plot_cv_iterations()
rd.plot_oscillation(size=2000)


"""