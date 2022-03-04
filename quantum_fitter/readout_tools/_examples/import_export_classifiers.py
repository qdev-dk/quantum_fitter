"""import quantum_fitter.readout_tools as rdt
import os

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, 'example_data/ss_q1_rabi_v_ampl_5_fine.hdf5')

# Set the instence 
rd = rdt.Readout(file_path, size=1000, verbose=1)
rd.do_fit()

# Exporting classifier
rd.export_classifier()

# Importing (1)
rd_test1 = rd.import_classifier()

# Importing (2)
filePath = 'example_data/ss_q1_rabi_v_ampl_5_fine.pickle'
rd_test2 = rdt.import_classifier(filePath)

# Testing
rd_test2.plot_classifier()"""