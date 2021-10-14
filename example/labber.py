import Labber
import quantum_fitter as qf
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

# Creat a test file in case of damaging
# copyfile('../_resonator/VNA_all_res_v_pow.hdf5', '../_resonator/VNA_all_res_v_pow_test.hdf5')

path = 'Z:/KianG/Resonator_Fitting_Test_Data/Al_on_sapphire_chip3/Power_sweeps_all_res_40dB_att_TWPA_on.hdf5'
t5 = qf.LabberData(filePath=path)
# qf.fit_all_labber_resonator(file_loc='C:/Users/Gao_z/Desktop/KU/ResonatorDataTest/Resonator_power_sweeps_40dBmatt_TWPA_on_2.hdf5',
#                             frequency=6.7333*1e9,
#                             power=list(range(-50, -40, 2)))

# freq, s21 = t5.fit_all_labber_resonator(repetition=2, power=list(range(-10, 0, 2)), frequency=[6.064*1e9])
# print(freq, s21)

# method with 'lc', 'aw', 'lcaw' or None
t5.pull_data(frequency=6.4972218e9, power=[-28])
t5.fit_data(model='ResonatorModel', method='lc', resonator_plot=True, window=0.05, sigma=0.15, verbose=1)
# t5.fit_data(model='ResonatorModel', resonator_plot=True, window=0.06, sigma=0.5, verbose=1)
plt.show()








# t5 = qf.LabberData('Z://Kian G//Resonator_power_sweeps_20dBm_attenuation.hdf5')
# t5.pull_data(power=[-50], frequency=[6.8447e9])
# t5.fit_data(model='ResonatorModel', resonator_plot=True, method='aw', window=0.05)
# plt.show()

# DVZ data
# import numpy as np
# po = -50
# freq, s21 = qf.read_dat('Z://Kian G//MBE_res_test_res9//rs_vna_S21_power_set_S21_frequency_set.dat', power=po)
# plt.plot(freq, np.abs(s21))
# plt.show()
# t5 = qf.QFit(freq, s21, model='ResonatorModel')
# t5.guess()
# t5.wash(method='linecomp', window=0.07)
# t5.add_weight(sigma=0.05)
# t5.do_fit()
# t5.polar_plot(power=po)
# plt.show()