# Quantum_fitter
To use the easy Qfit package, you need to first initialize an instance by ins = quantum_fitter.Qfit(x, y, model, params_init); 
Where params_init (can be empty) is a list (which is in sequence of your function parameters) or dict (with name: value) for initial value.
You can use your own model functions or build-in models in lmfit (accept by both list(str) or str).
[See more about lmfit build-in models, or all of them](https://lmfit.github.io/lmfit-py/builtin_models.html)

If you want to do advanced mod to change any parameters' properties, use ins.set_params() to alter.
```python
set_params(name: str, value: float=None, vary:bool=True, minimum=None, maximum=None, expression=None, brute_step=None):
```

Then, use ins.do_fit() to fit through lmfit.

If plot, use pretty_print(plot_settings), with following dict.

```python
plot_settings = {
    'x_label': 'Time (us)',
    'y_label': 'Voltage (mV)',
    'plot_title': 'datasource',
    'y_lim': [0, 80],
    'fit_color': 'C4'
    }
```
    
If print pdf, use **pdf_print**. if don't want figure output, use pdf_print(plot_settings)


##One example:##

==========================================
```python
import quantum_fitter as qf
from numpy import sqrt, pi, exp
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))

x = np.linspace(0, 10, 500)
y = gaussian(x, 8, 5, 0.6) + np.random.randn(500)

params_ini = {'intercept': 0,
              'slope': 0,
              'amplitude': 5,
              'center': 5,
              'sigma': 1}
# params_ini = [5, 5, 1]

a = qf.QFit(x, y, ['GaussianModel', 'LinearModel'], params_ini)
# a = qf.Qfit(x, y, gaussian)
# a.set_params('amp', 5)

a.do_fit()
# a.pretty_print()
a.pdf_print()
plt.show()
```
<img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_4996.GIF" width="30" height="30" />
<img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_5007.GIF" width="30" height="30" />
