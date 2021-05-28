# Quantum_fitter
<p align="center">
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=code-status&message=Good!&color=green"/>
  <img src="https://img.shields.io/static/v1?style=for-the-badge&label=initial-commit&message=Kian-Gao&color=inactive"/>
    <img src="https://img.shields.io/static/v1?style=for-the-badge&label=maintainer&message=Kian-Gao&color=inactive"/>
</p>

The purpose of this package is to provide a uniform, easy-to-use fitting protocol that everyone can share and extend, in the cQED-at-QDev group. It's highly flexible, and pretty easy to use. To use the `Qfit` package, you initialize an instance with your `x`-data, `y`-data, and the models you want to fit to as well as the initial guesses for parameters:
```python
qfit = quantum_fitter.Qfit(x, y, ['GaussianModel', 'LinearModel'], params_init); 
```
Where `params_init` can be a dict of the type `{'amplitude': initial_guess}` (or, in the case of a user-defined model, it can be a list, where the ordering reflects the order in which the user-defined model accepts parameters (see example below)), with initial values of the free parameters in the model. It can be an empty list, to let the fitter try to find initial values itself. An example of `params_init` for the `Gaussian + Linear`--model, initialized above:
```python
params_init = {'intercept': 0,
              'slope': 0,
              'amplitude': 5,
              'center': 5,
              'sigma': 1}
```
You can use your own model functions or build-in models in `lmfit`. The class instance accepts both a string (e.g. `GaussianModel`) or a list of strings (if you want to combine multiple models, i.e. a `GaussianModel` with a `LinearModel` on top).
[See more about builtin lmfit build-in models here](https://lmfit.github.io/lmfit-py/builtin_models.html)

If you want to change any parameters' properties, use `qfit.set_params()` to alter.
```python
# set_params(name: str, value: float=None, vary:bool=True, minimum=None, maximum=None, expression=None, brute_step=None)
qfit.set_params('amplitude', 5, maximum = 10)
```

If need to smooth the data beforehand, use `qfit.wash()` to [filter the spikes](https://docs.scipy.org/doc/scipy/reference/signal.html).
```python
qfit.wash(method='savgol', window_length=5, polyorder=1)
```

Then, use do_fit() to fit through lmfit:
```python 
qfit.do_fit()
```

If need plot, use `pretty_print()`:

```python
qfit.pretty_print()
```
    
To print the resulting plot to a pdf, use `pdf_print`:
```python
qfit.pdf_print('qfit.pdf')
```

To get the fit parameters' results, we can use `qfit.fit_params`, `qfit.err_params`. The method `qfit.fit_values` returns the y-data from the fit.

```python
f_p = qfit.fit_params() # return dictionary with all the fitting parameters 
f_e = qfit.err_params('amplitude') # return float of amplitude's fitting stderr
y_fit = qfit.fit_values()
```


Here is the built-in function example for `qfit`
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
<img src="https://github.com/cqed-at-qdev/quantum_fitter/blob/main/_test/qtest.png" width="400" height="300" />
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

# smooth the spikes
a.wash()

a.do_fit()
file_path = os.path.dirname(os.path.realpath(__file__))
a.pdf_print(file_path, 'qfit_test', plot_settings=plot_set)
```
Look into the test file, the template pdf file is there.
<img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_4996.GIF" width="30" height="30" /> <img src="https://github.com/gaozmm/Playground_gaozm/blob/main/QDev/IMG_5007.GIF" width="30" height="30" />

For more examples, please look into the example directory. 


# Appendix A: The build-in function list
**Peak-like models**, for more models, [tap here](https://lmfit.github.io/lmfit-py/builtin_models.html).

To obtain the parameters in the models, use ```qf.params('GaussianModel')``` to get.

[GaussianModel](https://lmfit.github.io/lmfit-py/builtin_models.html#gaussianmodel)

[LorentzianModel](https://lmfit.github.io/lmfit-py/builtin_models.html#lorentzianmodel)

[SplitLorentzianModel](https://lmfit.github.io/lmfit-py/builtin_models.html#splitlorentzianmodel)

[VoigtModel](https://lmfit.github.io/lmfit-py/builtin_models.html#voigtmodel)

[PseudoVoigtModel](https://lmfit.github.io/lmfit-py/builtin_models.html#pseudovoigtmodel)

[MoffatModel](https://lmfit.github.io/lmfit-py/builtin_models.html#moffatmodel)

[Pearson7Model](https://lmfit.github.io/lmfit-py/builtin_models.html#pearson7model)

[StudentsTModel](https://lmfit.github.io/lmfit-py/builtin_models.html#studentstmodel)

[BreitWignerModel](https://lmfit.github.io/lmfit-py/builtin_models.html#breitwignermodel)

[LognormalModel](https://lmfit.github.io/lmfit-py/builtin_models.html#lognormalmodel)

[DampedOscillatorModel](https://lmfit.github.io/lmfit-py/builtin_models.html#dampedoscillatormodel)

[DampedHarmonicOscillatorModel](https://lmfit.github.io/lmfit-py/builtin_models.html#dampedharmonicoscillatormodel)

[ExponentialGaussianModel](https://lmfit.github.io/lmfit-py/builtin_models.html#exponentialgaussianmodel)

[SkewedGaussianModel](https://lmfit.github.io/lmfit-py/builtin_models.html#skewedgaussianmodel)

[SkewedVoigtModel](https://lmfit.github.io/lmfit-py/builtin_models.html#skewedvoigtmodel)

[ThermalDistributionModel](https://lmfit.github.io/lmfit-py/builtin_models.html#thermaldistributionmodel)

[DoniachModel](https://lmfit.github.io/lmfit-py/builtin_models.html#doniachmodel)



