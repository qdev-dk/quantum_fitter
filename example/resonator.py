import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantum_fitter as qfit
import sys


# Data read from h5 file by Labber
##=====================================
# sys.path.append('Your directory/Labber/Script')
# import Labber
# datasource = './VNA_all_res_v_pow.hdf5'
# qubit = 3
# entry = 26
# dataChannel = 'VNA - S21'
# Lfile = Labber.LogFile(datasource)
# [xData, yData] = Lfile.getTraceXY(y_channel=dataChannel, entry=entry)
# freq = xData # Convert to µs unit
# S21 = yData

#=======================================
# Else data read from .dat file
freq, S21 = qfit.read_dat('../_resonator/MDC_25.dat', power=-40)
plt.plot(S21.real, S21.imag)

#=======================================

# Plot real and imag data
t5 = qfit.QFit(freq, S21, model='ResonatorModel')
t5.wash()
# Do a guess (Automatically store the guess parameters), and store the initial guess value for check
t5.guess()
t5.do_fit()
t5.polar_plot(power=-40)

fit_s21 = t5.fit_values()


