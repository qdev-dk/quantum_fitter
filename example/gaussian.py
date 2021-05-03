import quantum_fitter as qf
import numpy as np

def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


# Generate random number from gaussian distribution
x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) * 0.1

plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'datasource',
    'fit_color': 'C4',
    'fig_size': (8, 6),
}

gm = qf.QFit(x, y, gaussian)
gm.do_fit()
gm.pretty_print(plot_settings)
