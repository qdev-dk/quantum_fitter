import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantum_fitter as qfit

df = pd.read_csv('../_resonator/rs_vna_S21_power_set_S21_frequency_set.dat', delimiter='\t', header=0,
                 skiprows=lambda x: x in [0, 2])
df.columns = ['Power'] + list(df.columns[1:])
power = df['Power'].to_numpy()
freq = df['S21 frequency'].to_numpy()
mag = df['S21 magnitude'].to_numpy()
phase = df['S21 phase'].to_numpy()
# rq.load_polardata(32, path='../_resonator')
S21 = mag * np.exp(1j * phase)
# _, popt, pcov, _, _ = run_feedline(freq, S21)
# f = np.vstack((freq, power))

# rq.plot_quality(f, S21)

from quantum_fitter import _model

t5 = qfit.QFit(freq, S21, model='ResonatorModel')
t5.guess()
guess_s21 = t5.eval(x=freq)
t5.print_params()
plt.figure()
plt.scatter(freq, 20*np.log10(np.abs(guess_s21)), c='r', s=0.5)
plt.show()

t5.do_fit()
measured_s21 = t5.fit_values()


plt.figure()
plt.plot(freq, 20*np.log10(np.abs(measured_s21)))
plt.ylabel('|S21| (dB)')
plt.xlabel('MHz')
plt.title('simulated measurement')
plt.show()

