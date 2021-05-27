import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantum_fitter as qfit
#===============================
# Reading From labber

import sys
sys.path.append('D:/Labber/Script')
import Labber

# datasource = './VNA_all_res_v_pow.hdf5'
# qubit = 3
# entry = 26
# dataChannel = 'VNA - S21'
# Lfile = Labber.LogFile(datasource)
# [xData, yData] = Lfile.getTraceXY(y_channel=dataChannel, entry=entry)
# freq = xData # Convert to µs unit
# S21 = yData
#===============================
df = pd.read_csv('../_resonator/MDC_25.dat', delimiter='\t', header=0,
                 skiprows=lambda x: x in [0, 2])
# df = pd.read_csv('../_resonator/rs_vna_S21_power_set_S21_frequency_set.dat', delimiter='\t', header=0,
#                  skiprows=lambda x: x in [0, 2])
df.columns = ['Power'] + list(df.columns[1:])
power = df['Power'].to_numpy()
power_5 = np.argwhere(power == -40)
freq = df['S21 frequency'].to_numpy()[power_5]
mag = df['S21 magnitude'].to_numpy()[power_5]
phase = df['S21 phase'].to_numpy()[power_5]
S21 = mag * np.exp(1j * phase)
plt.plot(S21.real, S21.imag)
plt.show()

# # rq.load_polardata(32, path='../_resonator')
# # _, popt, pcov, _, _ = run_feedline(freq, S21)
# # f = np.vstack((freq, power))
#
# # rq.plot_quality(f, S21)
#
# from quantum_fitter import _model
#==============================================
# t5 = qfit.QFit(freq, S21, model='ComplexResonatorModel')
# t5.guess()
# guess_s21 = t5.eval(x=freq)
# plt.figure()
# plt.scatter(freq+1e-6*freq, np.abs(guess_s21), c='r', s=0.1)
# plt.scatter(freq+1e-6*freq, np.abs(S21), c='b', s=0.1)
# plt.show()
# plt.figure()
# t5.do_fit()
# s21_fit = t5.fit_values()
# # plt.scatter(freq, 20*np.log10(np.abs()), c='b', s=0.3)
# plt.ylabel('|S21| (dB)')
# plt.xlabel('MHz')
# plt.title('simulated measurement')
# plt.scatter(freq+1e-6*freq, np.abs(S21), c='r', s=0.1)
# plt.scatter(freq+1e-6*freq, np.abs(s21_fit), c='b', s=0.1)
# plt.show()
#=========================================
def plot_ri(data, *args, **kwargs):
    plt.plot(data.real, data.imag, *args, **kwargs)

freq = freq * 1E-6
freq = freq - freq[int(len(freq)/2)]
t5 = qfit.QFit(freq, S21, model='ResonatorModel')
t5.wash()
t5.guess()
guess_s21 = t5.eval(x=freq)
t5.print_params()
t5.do_fit()
fit_s21 = t5.fit_values()
plt.figure()
plot_ri(S21, '.')
plot_ri(fit_s21, 'r.-', label='best fit')
plot_ri(guess_s21, 'k--', label='inital fit')
plt.legend(loc='best')
plt.xlabel('Re(S21)')
plt.ylabel('Im(S21)')

plt.figure()
plt.plot(freq, 20*np.log10(np.abs(S21)), '.')
plt.plot(freq, 20*np.log10(np.abs(fit_s21)), 'r.-', label='best fit')
plt.plot(freq, 20*np.log10(np.abs(guess_s21)), 'k--', label='initial fit')
plt.legend(loc='best')
plt.ylabel('|S21| (dB)')
plt.xlabel('MHz')
plt.show()
