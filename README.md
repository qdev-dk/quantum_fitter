# Quantum_fitter
The purpose of making this package is to shorten the time we deal with the data, and make it easier to analyse. To use the easy Qfit package, you need to first initialize an instance by 
```python
qfit = quantum_fitter.Qfit(x, y, ['GaussianModel', 'LinearModel'], params_init); 
```
Where params_init (can be empty) is a list (which is in sequence of your function parameters) or dict (with name: value) for initial value.
```python
params_ini = {'intercept': 0,
              'slope': 0,
              'amplitude': 5,
              'center': 5,
              'sigma': 1}
```
You can use your own model functions or build-in models in lmfit (accept by both list(str) or str).
[See more about lmfit build-in models, or all of them](https://lmfit.github.io/lmfit-py/builtin_models.html)

If you want to do advanced mod to change any parameters' properties, use qfit.set_params() to alter.
```python
#set_params(name: str, value: float=None, vary:bool=True, minimum=None, maximum=None, expression=None, brute_step=None)
qfit.set_params('amplitude', 5, maximum = 10)
```

Then, use do_fit() to fit through lmfit.
```python 
qfit.do_fit()
```

If need plot, use pretty_print

```python
qfit.pretty_plot()
```
    
If print pdf, use **pdf_print**.
```python
qfit.pdf_print('qfit.pdf')
```

To get the fit parameters' results, we can use qfit.fit_params, qfit.err_params. Also qfit.fit_values to get the final fitting data.

```python
f_p = qfit.fit_params() # return dictionary with all the fitting parameters 
f_e = qfit.err_params('amplitude') # return float of amplitude's fitting stderr
y_fit = qfit.fit_values()
```

Here is the built-in function example for qfit
## Example ##

```python
import quantum_fitter as qf
from numpy import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) + 0.2 * x + 0.1

params_ini = {'intercept': 0,
              'slope': 0,
              'amplitude': 5,
              'center': 5,
              'sigma': 1}

qfit = qf.QFit(x, y, ['GaussianModel', 'LinearModel'], params_ini)
qfit.set_params('amp', value=5, minimum=-100, maximum=100)

qfit.do_fit()

              
plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'Maxwell's demon',
    'y_lim': [0, 8],
    'fit_color': 'C4',
    'fig_size': (8, 6),
}

qfit.pretty_print()
qfit.pdf_print('./fit_pdf/qfit.pdf')
plt.show()
```
<p align="center">
<img src="https://github.com/cqed-at-qdev/quantum_fitter/blob/main/test/qtest.png" width="400" height="300" />
</p>

Or we can use our own modification function

```python
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500)

params_ini = {'amplitude': 5,
              'center': 5,
              'sigma': 1}
# Alternatively 
# params_ini = [5, 5, 1]

qfit = qf.QFit(x, y, gaussian, params_ini)

a.do_fit()
file_path = os.path.dirname(os.path.realpath(__file__))
a.pdf_print(file_path, 'qfit_test', plot_settings=plot_set)
```
Look into the test file, the template pdf file is there.
<img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_4996.GIF" width="30" height="30" /> <img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_5007.GIF" width="30" height="30" />

