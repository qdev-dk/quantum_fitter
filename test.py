import sys
import quantum_fitter as qf
from numpy import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500)
params_ini = [5, 5, 1]

a = qf.QFit(x, y, gaussian, params_ini)
a.set_params('amp', 5)
a.do_fit()

a.pretty_print()
plt.show()
