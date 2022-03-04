"""import quantum_fitter.readout_tools as rdt
import os

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')

# Set the instence 
rd = rdt.Readout(file_path, size=1000, verbose=1)

# Set paramaters
rd.set_cv_search(mode='random', aggressive_elimination=True)
for kernel in ['rbf', 'sigmoid', 'linear', 'poly']:
    rd.set_cv_params(params={'classifier__kernel': [kernel]})
    
    rd.do_fit() # run fit for every loop
    rd.plot_classifier() # plots classifier
    rd.cal_expectation_values(size=2000) # calculates expectation values for classifier
    
rd.plot_oscillation(size=2000, mode='expectation') # plots expectation values for all classifiers
rd.plot_oscillation(size=2000) # plots probability values for all classifiers
"""