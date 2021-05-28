import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantum_fitter as qfit
import sys


# Data read from h5 file by Labber
##=====================================
sys.path.append('D:/Labber/Script')
import Labber
datasource = '../_resonator/VNA_all_res_v_pow.hdf5'
qubit = 3
ent = range(142, 150, 31)
pow = -44

#=======================================
# Else data read from .dat file
# freq, S21 = qfit.read_dat('../_resonator/MDC_25.dat', power=pow)

#=======================================
# Better to move the center of frequency to 0
for entry in ent:
    dataChannel = 'VNA - S21'
    Lfile = Labber.LogFile(datasource)
    [xData, yData] = Lfile.getTraceXY(y_channel=dataChannel, entry=entry)
    freq = xData*1e-9 # Convert to µs unit
    S21 = yData
    # S21 -= np.mean(S21)
    # f0 = freq[int(len(freq)/2)]
    # freq = freq - f0

    t5 = qfit.QFit(freq, S21, model='ResonatorModel',)
    t5.wash(method='savgol', window_length=5, polyorder=2)
    # Do a guess (Automatically store the guess parameters), and store the initial guess value for check
    t5.guess()
    t5.eval(x=freq)
    t5.do_fit()
    t5.polar_plot(power=pow, id=entry, plot_settings={'plot_guess': 0})

    fit_s21 = t5.fit_values()
plt.show()


