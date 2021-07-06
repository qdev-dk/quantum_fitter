import Labber
import quantum_fitter as qf
import sys
from shutil import copyfile
import matplotlib.pyplot as plt

# Creat a test file in case of damaging
# copyfile('../_resonator/VNA_all_res_v_pow.hdf5', '../_resonator/VNA_all_res_v_pow_test.hdf5')


t5 = qf.LabberData('Z://Kian G//Resonator_power_sweeps_20dBm_attenuation.hdf5')
t5.pull_data(repetition=3, power=[-2], frequency=[6.8447e9])
t5.fit_data(model='ResonatorModel', resonator_plot=True, method='lc')
plt.show()
