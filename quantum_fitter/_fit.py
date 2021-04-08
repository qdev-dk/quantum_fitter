from lmfit.models import LorentzianModel, GaussianModel, ExponentialModel
import numpy as np
"""
To use the easy Qfit package, you need to first initialize an instance by ins = quantum_fitter.CreateFit(x, y)
Then, use ins.do_fit(model) to fit through lmfit. If you want to change any parameters, 



"""

class CreateFit:
    def __init__(self, data_x, data_y):
        self._datax = data_x
        self._datay = data_y
        self._params_dict = {}
        self._frequency, self._phase, self._upperBound, self._lowerBound, self.

    def set_conds(self, value_name: str, initial_value=None, fixed=False,
                  frequency=None, phase=None, up_bound=None, lo_bound=None):
        self.freq = 1


    def do_fit(self, model):

    def fit_through_lm

