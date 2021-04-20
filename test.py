import quantum_fitter as qf
from numpy import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) + 0.2 * x + 0.1

params_ini = {'intercept': 0,
              'slope': 0,
              'amplitude': 5,
              'center': 5,
              'sigma': 1}

plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'datasource',
    'y_lim': [0, 8],
    'fit_color': 'C4',
    'fig_size': (8, 6),
}

a = qf.QFit(x, y, ['GaussianModel', 'LinearModel'], params_ini)
# a.set_params('amp', 5)

a.do_fit()
a.pretty_print(plot_settings)
a.pdf_print()
params = a.fit_params()
plt.show()
