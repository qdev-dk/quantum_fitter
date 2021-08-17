import quantum_fitter as qf
import numpy as np

t2 = qf.LabberData('Z:/Kian G/2021_08_17/MIT_ZI__Q1_FF_T1_5.hdf5')
freq, s21 = t2.pull_data(mode='rabi')
t2.fit_data(verbose=1)

# import matplotlib.pyplot as plt
# x = np.linspace(0, 1, 100)
# y = -1 * np.exp(-x * 5)
# plt.plot(x, y)
# plt.show()