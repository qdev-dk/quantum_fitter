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



## Example ##
An example of using 'qfit' can be found here: 
[Example](https://qdev-dk.github.io/quantum_fitter/example_notebooks/gaussian_fitting.html)

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

## Installation

On a good day, installation is as easy as
```
$ git clone https://github.com/qdev-dk/quantum_fitter.git
$ cd quantum_fitter
$ pip install .
```

## Running the tests

If you have gotten 'quantum_fitter' from source, you may run the tests locally.

Install `quantum_fitter` along with its test dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r test_requirements.txt
```

Then run `pytest` in the `tests` folder.

## Building the documentation

If you have gotten `quantum_fitter` from source, you may build the docs locally.

Install `quantum_fitter` along with its documentation dependencies into your virtual environment by executing the following in the root folder

```bash
$ pip install .
$ pip install -r docs_requirements.txt
```

You also need to install `pandoc`. If you are using `conda`, that can be achieved by

```bash
$ conda install pandoc
```
else, see [here](https://pandoc.org/installing.html) for pandoc's installation instructions.

Then run `make html` in the `docs` folder. The next time you build the documentation, remember to run `make clean` before you run `make html`.
