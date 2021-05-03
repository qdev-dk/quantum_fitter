import quantum_fitter as qf
import numpy as np

def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


# Generate random number from gaussian distribution
x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) + 0.2 * x + 0.1
model = ['GaussianModel', 'LinearModel']

plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'datasource',
    'fit_color': 'C4',
    'fig_size': (8, 6),
}

gm = qf.QFit(x, y, model)
gm.set_params('amp', value=0, vary=True, minimum=-10)
gm.do_fit()
gm.pdf_print('C:/qfpdf_save', 'Linear fitting', plot_settings)
