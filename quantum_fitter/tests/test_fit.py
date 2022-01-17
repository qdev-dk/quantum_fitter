import pytest
import quantum_fitter as qf
import numpy as np

def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

def test_lmfit_model():
    # Generate random number from gaussian distribution
    x = np.linspace(0, 10, 500)
    y = gaussian(x, 8, 5, 0.6) + np.random.randn(500) * 0.001

    params_ini = {'amplitude': 8.0,
                  'center': 5.0,
                  'sigma': 0.6}

    qfit = qf.QFit(x, y,'GaussianModel', params_ini)
    qfit.do_fit()

    fit_params = qfit.fit_params()

    for key in params_ini.keys():
        assert fit_params[key] == pytest.approx(params_ini[key], 0.1)

def test_list_of_moddels():
    x = np.linspace(0, 10, 500)
    y = gaussian(x, 8, 5, 0.6) + 0.5*x + 1.0 + np.random.randn(500) * 0.001
    
    params_ini = {'amp': 8, 'cen': 5, 'wid': 0.6, 'slope': 0.5, 'intercept': 1.0}
    
    qfit = qf.QFit(x, y,[gaussian,'LinearModel'],params_ini)
    qfit.do_fit()
    fit_params = qfit.fit_params()

    for key in params_ini.keys():
        assert fit_params[key] == pytest.approx(params_ini[key], 0.1)


def test_prefix_same_model():
    x = np.linspace(0, 10, 500)
    y = gaussian(x, 8, 3, 1) +  gaussian(x, 8, 8, 1) +  gaussian(x, 8, 5, 1) + np.random.randn(500) * 0.001
    params_ini = {'amplitude': 8.0,
              'center': 3.0,
              'sigma': 1.0,
              'f2_amplitude': 8.0,
              'f2_center': 8.0,
              'f2_sigma': 1.0,
              'f3_amplitude': 8.0,
              'f3_center': 5.0,
              'f3_sigma': 1.0}

    qfit = qf.QFit(x, y,['GaussianModel','GaussianModel','GaussianModel'],params_ini)
    qfit.do_fit()
    fit_params = qfit.fit_params()

    for key in params_ini.keys():
        assert fit_params[key] == pytest.approx(params_ini[key], 0.1)
