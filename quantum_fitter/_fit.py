from lmfit import Model, Minimizer, Parameters, report_fit, models
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import os
import quantum_fitter._model as md


class QFit:
    def __init__(self, data_x, data_y=None, model=None, params_init=None, method='leastsq', **kwargs):
        self._datax = data_x.flatten()
        self._datay, self._raw_y, self._fity = data_y, data_y, None
        # define the history of y, use for pdf or plots
        if self._datay is not None:
            self._datay, self._fity = data_y.flatten(), np.empty((0, len(data_y)))
            self._origin_data = self._datay
        self.result = 0
        self.method = method
        self._fig, self._ax = 0, 0
        self.wash_status = False
        self._init_guess_y = None

        # using default build-in model in lmfit. If want to do multiple build-in model, just pass in a list of str
        # Example: model=['LinearModel', 'LorentzianModel']
        if isinstance(model, str):
            # Detect if we are using resonator model
            if model in ['ComplexResonatorModel', 'ResonatorModel']:
                self._qmodel = getattr(md, model)()
                self._params = self._qmodel.make_params()
            else:
                self._qmodel = getattr(models, model)()
                self._params = self._qmodel.make_params()

        elif isinstance(model, list) or isinstance(model, set):
            self._qmodel = getattr(models, model[0])()
            self._params = self._qmodel.make_params()
            if len(model) > 1:
                for m in model[1:]:
                    self.add_models(m)
        else:
            self._qmodel = Model(model)
            self._params = self._qmodel.make_params()

        # set initial value by using either list (in sequence of params) or dict (with name keys and value items)
        if isinstance(params_init, list):
            for n_params in range(len(params_init)):
                self._params.add(self._qmodel.param_names[n_params], params_init[n_params])
        elif isinstance(params_init, dict):
            for para_name in params_init.keys():
                self._params.add(para_name, params_init[para_name])

        self.x_name = self._qmodel.param_names[0]

    def __str__(self):
        return 'Lmfit hi'

    def set_params(self, name: str, value: float = None, vary: bool = True, minimum=None, maximum=None, expression=None
                   , brute_step=None):
        self._params.add(name, value, vary, minimum, maximum, expression, brute_step)

    @property
    def params(self):
        '''
        Set or get the parameters of current models.
        :return: Parameters' dictionary
        '''
        print(self._params.valuesdict())
        return self._params.valuesdict()

    @params.setter
    def params(self, init_dict: dict):
        for para_name in init_dict.keys():
            self._params.add(para_name, init_dict[para_name])

    @property
    def data_y(self):
        if not self._datay:
            print('No data y now!')
            return
        return self._datay

    @data_y.setter
    def data_y(self, data_y):
        self._datay = data_y

    def make_params(self, **kwargs):
        return self._qmodel.make_params(**kwargs)

    def add_models(self, model, merge: str = '+'):
        '''
        Add model to current models.
        :param model: The model you want to add in.
        :param merge: The operation needed to merge, can be +,-,/,*.
        :return:
        '''
        if isinstance(model, str):
            _new_model = getattr(models, model)()
        else:
            _new_model = Model(model)
        # Check if there is any same parameter name
        name_list = set(self._qmodel.param_names).intersection(_new_model.param_names)
        if name_list:
            print('The build-in models have the same parameter name' + str(name_list))
            if isinstance(model, str):
                model_name = model
                prefix = ''.join([c for c in model if c.isupper()])
            else:
                prefix = model.__name__
                model_name = prefix
            _new_model.prefix = prefix
            print('Add prefix', prefix, 'to the parameters in', model_name)

        if merge == '+':
            self._qmodel += _new_model
        elif merge == '*':
            self._qmodel *= _new_model
        elif merge == '-':
            self._qmodel -= _new_model
        elif merge == '/':
            self._qmodel /= _new_model
        else:
            self._qmodel += _new_model
            print('Merge style wrongly specified. Using \'+\' operator instead\n')
        self._params += _new_model.make_params()

    def eval(self, params=None, **kwargs):
        if params is None:
            self._init_guess_y = self._qmodel.eval(self._params, **kwargs)
            return self._init_guess_y
        else:
            self._init_guess_y = self._qmodel.eval(params, **kwargs)
            return self._init_guess_y

    def fit_params(self, name: str = None):
        if name is None:
            return self.result.values
        return self.result.values[name]

    def guess(self):
        self._params = self._qmodel.guess(self._datay, f=self._datax)

    def err_params(self, name: str = None):
        if name is None:
            return self._params_stderr()
        return self._params_stderr()[name]

    def fit_values(self):
        return self.result.best_fit

    def do_fit(self):
        self.result = self._qmodel.fit(self._datay, self._params, x=self._datax, method=self.method)
        print(self.result.fit_report())
        self._fity = self.result.best_fit

    def wash(self, method='savgol', **kwargs):
        if method == 'savgol':
            _win_l = kwargs.get('window_length') if kwargs.get('window_length') else 3
            _po = kwargs.get('polyorder') if kwargs.get('polyorder') else 2
            if np.iscomplexobj(self._datay):
                rr = savgol_filter(np.real(self._datay), _win_l, _po)
                ri = savgol_filter(np.imag(self._datay), _win_l, _po)
                self._datay = np.vectorize(complex)(rr, ri)
            else:
                self._datay = savgol_filter(self._datay, _win_l, _po)

    def pretty_print(self, plot_settings=None):
        '''Basic function for plotting the result of a fit'''
        fit_params, error_params, fit_value = self.result.best_values, self._params_stderr(), \
                                              self.result.best_fit.flatten()
        _fig_size = (8, 6) if plot_settings is None else plot_settings.get('fig_size', (8, 6))
        self._fig, ax = plt.subplots(1, 1, figsize=_fig_size)
        # Add the original data
        data_color = 'C0' if plot_settings is None else plot_settings.get('data_color', 'C0')
        ax.plot(self._datax, self._datay, '.', label='Data', color=data_color, markersize=10, zorder=10)
        fit_color = 'gray' if plot_settings is None else plot_settings.get('fit_color', 'k')
        # Add fitting curve:
        ax.plot(self._datax, fit_value, '-', linewidth=1, label='Fit', color=fit_color)
        ax.plot(self._datax, fit_value, 'o', markersize=3, color=fit_color)
        # Hack to add legend with fit-params:
        for key in fit_params.keys():
            ax.plot(self._datax[0], fit_value[0], 'o', markersize=0,
                    label='{}: {:4.4}±{:4.4}'.format(key, fit_params[key], error_params[key]))
        # Rescale plot if user wants it:
        if plot_settings is not None:
            ax.set_xlabel(plot_settings.get('x_label', 'x_label not set'))
            ax.set_ylabel(plot_settings.get('y_label', 'y_label not set'))
            if 'x_lim' in plot_settings.keys():
                ax.set_xlim(plot_settings['x_lim'])
            if 'y_lim' in plot_settings.keys():
                ax.set_ylim(plot_settings['y_lim'])
        ax.legend()
        title = 'Data source not given' if plot_settings is None else plot_settings.get('plot_title', 'no name given')
        ax.set_title('Datasource: ' + title + '\n Fit type: ' + str(self._qmodel.name))
        # Check if use wants figure and return if needed:
        if plot_settings is None or plot_settings.get('show_fig', None) is None:
            plt.show()

        if plot_settings is not None and plot_settings.get('return_fig', None) is not None:
            return self._fig, ax

    def polar_plot(self, plot_settings={}, power=None, f0=None, id=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        ax1.plot(self._fity.real, self._fity.imag, 'r', label='best fit', linewidth=1.5)
        ax1.scatter(self._raw_y.real, self._raw_y.imag, c='grey', s=1)
        ax1.set_title('Raw S21 Complex Plane', fontdict={'fontsize': 10})
        ax1.set_xlabel('Re(S21)')
        ax1.set_ylabel('Im(S21)')

        ax2.plot(self._datax, 20*np.log10(np.abs(self._fity)), 'r', label='best fit', linewidth=1.5)
        ax2.scatter(self._datax, 20*np.log10(np.abs(self._raw_y)), c='grey', s=1)
        ax2.set_title('S21 Mag', fontdict={'fontsize': 10})
        ax2.set_xlabel('Frequency / GHz')
        ax2.set_ylabel('S21(dB)')

        ax3.plot(self._datax, np.angle(self._fity), 'r', label='best fit', linewidth=1.5)
        ax3.scatter(self._datax, np.angle(self._raw_y), c='grey', s=1)
        ax3.set_title('S21 Phase', fontdict={'fontsize': 10})
        ax3.set_xlabel('Frequency / GHz')
        ax3.set_ylabel('Angle / rad')

        fit_info = '$Q_{int}= $'+str("{0:.1f}".format(self.fit_params('Qi')*1e3))+'    '
        fit_info += '$Q_{ext}= $'+str("{0:.1f}".format(self.fit_params('Qe_mag')*1e3))

        if power:
            fit_info += '    ' + '$P_{VNA}=$ ' + str(power) + 'dBm'
        if f0:
            fit_info += '    ' + 'f0= ' + str("{0:.1f}".format(f0)) + 'MHZ'
        if id:
            fit_info += '    ' + 'id= ' + str(id)

        fig.suptitle(fit_info, fontdict={'fontsize': 10})


        if plot_settings.get('plot_guess', None) is not None:
            ax1.plot(self._init_guess_y.real, self._init_guess_y.imag, 'k--', label='inital fit')

        fig.tight_layout()
        # plt.show()

    def pdf_print(self, file_dir, filename, plot_settings=None):
        import datetime
        from matplotlib.backends.backend_pdf import PdfPages

        os.chdir(file_dir)
        # Create the PdfPages object to which we will save the pages:
        with PdfPages(filename) as pdf:
            # if don't wanna output figure, only want pdf pages, do it here.
            if self._fig == 0:
                self.pretty_print(plot_settings=plot_settings)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Qfit PDF Example'
            d['Author'] = 'Kian'
            d['Subject'] = 'Qfit'
            d['CreationDate'] = datetime.datetime.today()
            d['ModDate'] = datetime.datetime.today()
        pass

    def print_params(self):
        self._params.pretty_print()

    def _params_stderr(self):
        stderr = {}
        for param in self.result.params.keys():
            stderr[param] = self.result.params.get(params).stderr
        return stderr

    def _init_params(self):
        return self._params.valuesdict()



def params(name: str):
    if name in ['ComplexResonatorModel', 'ResonatorModel']:
        _params=getattr(md, name)().param_names
        print(name + '\'s parameters: ' + str(_params))
        return _params
    _params = getattr(models, name)().param_names
    print(name + '\'s parameters: ' + str(_params))
    return _params


def read_dat(file_location: str, power):
    import pandas as pd
    df = pd.read_csv(file_location, delimiter='\t', header=0,
                     skiprows=lambda x: x in [0, 2])
    df.columns = ['Power'] + list(df.columns[1:])
    _power = df['Power'].to_numpy()
    _power_mask = np.argwhere(_power == power)  # Choose the power you fit
    freq = df['S21 frequency'].to_numpy()[_power_mask]
    print(freq)
    mag = df['S21 magnitude'].to_numpy()[_power_mask]
    phase = df['S21 phase'].to_numpy()[_power_mask]
    S21 = mag * np.exp(1j * phase)

    # Scale and centerize(?)the frequency
    freq = freq * 1e-9

    return freq, S21


