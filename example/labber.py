import Labber
import quantum_fitter as qf
import sys
from shutil import copyfile
import matplotlib.pyplot as plt

# Creat a test file in case of damaging
# copyfile('../_resonator/VNA_all_res_v_pow.hdf5', '../_resonator/VNA_all_res_v_pow_test.hdf5')


t5 = qf.LabberData('../_resonator/Resonator_power_sweeps_20dBm_attenuation _for_fitting.hdf5')
t5.pull_data(repetition=[2], power=[-30, -24], frequency=[5.36641*1e9, 5.41507*1e9])
# t5.fit_data()
# t5.polar_plot()
# t5.push_data()
# plt.show()
