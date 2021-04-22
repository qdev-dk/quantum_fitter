from numpy import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt
import sys
import quantum_fitter as qf
from lmfit import models

sys.path.append('D:/Labber/Script')
import Labber

datasource = './BFC3-16-Qu3_T1_quick.hdf5'
qubit = 3
entry = 4
dataChannel = 'MQ PulseGen - Voltage, QB'+str(qubit)
Lfile = Labber.LogFile(datasource)
[xData, yData] = Lfile.getTraceXY(y_channel=dataChannel, entry=entry)
xData = xData*1e6 # Convert to µs unit
yData = np.imag(yData)

def sin_func(x, amp, freq, shift):
    return amp * np.sin(freq * x + shift)

# sin_func = sin_func()


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def sin_func(x, amplitde, freq, shift):
    return amplitde * np.sin(freq * x + shift)


x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) + 0.2 * x + 0.1
params_ini = {'amp': 5,
              'amplitude': -0.01,
              'decay': 10,
              'freq': 10,
              'shift': 1.5}

plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'datasource',
    'fit_color': 'C4',
    'fig_size': (8, 6),
}


a = qf.QFit(xData, yData, ['ExponentialModel'])
a.wash(window_length=5, polyorder=1)
# a.params = params_ini

# a.set_params('amp', 5)

a.do_fit()
a.pretty_print(plot_settings)
# file_path = os.path.dirname(os.path.realpath(__file__))
# a.pdf_print(file_path, 'qfit_test')
params = a.fit_params()
plt.show()