from lmfit import Model, Minimizer, Parameters, report_fit, models
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import os
import quantum_fitter._model as md


class QFit:
    def __init__(self, data_x, data_y=None, model=None, params_init=None, method='leastsq', **kwargs):
        self._raw_y = data_y.flatten()
        # data_y /= np.mean(np.abs(data_y)[[0, -1]])
        self._datax = data_x.flatten()
        self._datay, self._fity = data_y, None
        # define the history of y, use for pdf or plots
        if self._datay is not None:
            self._datay, self._fity = data_y.flatten(), np.empty((0, len(data_y)))
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
        self.weight = None
        self.wash_params = None

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
        self.eval(x=self._datax)

    def filter_guess(self, level=5):
        """
        The basic idea is to do the smoothing first, and then fit with the smooth data to get a decent initial guess.
        :return:
        Robust initial guess
        """
        self.guess()
        self.wash(method='savgol', window_length=level, polyorder=3)

        # Do fit with the heavy filter result
        _result = self._qmodel.fit(self._datay, self._params, x=self._datax, method=self.method)
        self._params = _result.params
        self._datay = self._raw_y
        self.eval(x=self._datax)

    def err_params(self, name: str = None):
        if name is None:
            return self._params_stderr()
        return self._params_stderr()[name]

    def fit_values(self):
        return self.result.best_fit

    def add_weight(self, array=None, mode='resonator', sigma=0.1):
        if array is None:
            weight_x = np.linspace(-1, 1, len(self._datax))
            _sigma = sigma
            muu = weight_x[np.argmin(np.log10(self._datay))]
            self.weight = np.exp(-((weight_x-muu)**2 / (2.0 * _sigma**2)))
            self.weight = (self.weight + 0.1) / 1.1


    def do_fit(self, verbose=None):
        self.result = self._qmodel.fit(self._datay, self._params, x=self._datax, method=self.method, weights=self.weight)
        # self._params = self.result.params
        if verbose:
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

        if method == 'cut':
            _win = kwargs.get('window') if kwargs.get('window') else [1/3, 2/3]
            self._datax = self._datax[int(len(self._datax) * _win[0]):int(len(self._datax) * _win[1])]
            self._datay = self._datay[int(len(self._datay) * _win[0]):int(len(self._datay) * _win[1])]
            self._fity = self._fity[int(len(self._fity) * _win[0]):int(len(self._fity) * _win[1])]
            self._raw_y = self._raw_y[int(len(self._raw_y) * _win[0]):int(len(self._raw_y) * _win[1])]

        if method == 'linearcomp':
            s = (self._datay[-1] - self._datay[0]) / (self._datax[-1] - self._datax[0])
            c = self._datay[0] - s * self._datax[0]
            background = s * self._datax + c
            self._datay -= background
            # plt.plot(self._datax, background.real, '--', label='breal')
            # plt.plot(self._datax, background.imag, '--', label='bimag')
            # plt.legend()
            # plt.plot(self._datax, self._raw_y)
            # A = np.mean(np.abs(self._datay)[[0, -1]])
            # self._datay = self._datay / A
            # self._raw_y = self._raw_y / A

        if method == 'complexcomp':
            # inspired and from David's code
            _window = int(kwargs.get('window') * len(self._datax)) \
                if kwargs.get('window') else int(0.06 * len(self._datax))
            phase = np.unwrap(np.angle(self._datay))
            line_fit = np.polyfit([np.mean(self._datax[:_window]), np.mean(self._datax[-_window:])],
                                [np.mean(phase[:_window]), np.mean(phase[-_window:])], 1)

            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax).real, label='lcreal')
            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax).imag, label='lcimag')
            # plt.plot(self._datax, self._datay.imag)
            self._datay = self._datay * np.exp(-1j * line_fit[0] * self._datax)
            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax))
            # plt.plot(self._datax, self._datay.imag, ls='--', c='grey')
            self.wash_params = [line_fit[0]]


    def cov_mat(self):
        return self.result.covar

    def plot_cov_mat(self):
        f = plt.figure(figsize=(8, 6))
        _cov_mat = self.cov_mat()
        _tick = self._params_name()
        print(_cov_mat)
        plt.matshow(_cov_mat, fignum=f.number)
        plt.xticks(range(len(_tick)), _tick, fontsize=10,
                   rotation=45)
        plt.yticks(range(len(_tick)), _tick, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.title('Correlation Matrix', fontsize=11)

    def pretty_print(self, plot_settings=None):
        '''Basic function for plotting the result of a fit'''
        plt.figure()
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

    def polar_plot(self, plot_settings={}, power=99999, f0=None, id=None, suptitle=''):
        angle = 1
        if self.wash_params:
            self._fity = self._fity * np.exp(1j * self.wash_params[0] * self._datax)
            # angle = np.exp(-1j * (self._datax * self.result.values['phi1'] + self.result.values['phi2']))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
        ax1.plot(self._fity.real, self._fity.imag, 'r', label='best fit', linewidth=1.5)
        ax1.scatter(self._raw_y.real, self._raw_y.imag, c='grey', s=1)
        ax1.set_title('Raw S21 Complex Plane', fontdict={'fontsize': 10})
        ax1.set_xlabel('Re(S21)')
        ax1.set_ylabel('Im(S21)')

        ax2.plot(self._datax, 20*np.log10(np.abs(self._fity)), 'r', label='best fit', linewidth=1.5)
        ax2.scatter(self._datax, 20*np.log10(np.abs(self._raw_y)), c='grey', s=1)
        ax2.scatter(self._datax[np.argmin(20*np.log10(np.abs(self._raw_y)))], np.min(20*np.log10(np.abs(self._raw_y)))
                    , c='b', s=5)
        ax2.set_title('S21 Mag', fontdict={'fontsize': 10})
        ax2.set_xlabel('Frequency / GHz')
        ax2.set_ylabel('S21(dB)')

        ax3.plot(self._datax, np.angle(self._fity*angle), 'r', label='best fit', linewidth=1.5)
        ax3.scatter(self._datax, np.angle(self._raw_y*angle), c='grey', s=1)
        ax3.set_title('S21 Phase', fontdict={'fontsize': 10})
        ax3.set_xlabel('Frequency / GHz')
        ax3.set_ylabel('Angle / rad')

        # if self._qmodel.name == 'ResonatorModel':
        fit_info = '$Q_{int}= $'+str("{0:.1f}".format(self.fit_params('Qi')*1e3))+'    '
        fit_info += '$Q_{ext}= $'+str("{0:.1f}".format(self.fit_params('Qe_mag')*1e3))
        # if self._qmodel.name == 'ResonatorModel':
        #     Qe = self.fit_params('Q_e_real') + 1j * self.fit_params('Q_e_imag')
        #     Qi = 1 / (1 / self.fit_params('Q') - 1 / self.fit_params('Q_e_real')) * 1e3
        #     fit_info = '$Q_{int}= $' + str("{0:.1f}".format(self.fit_params('Qi') * 1e3)) + '    '
        #     fit_info += '$Q_{ext}= $' + str("{0:.1f}".format(self.fit_params('Qe_mag') * 1e3))

        if power != 99999:
            fit_info += '    ' + '$P_{VNA}=$ ' + str(power) + 'dBm'
        if f0:
            fit_info += '    ' + 'f0= ' + str("{0:.4f}".format(f0)) + 'GHZ'
        if id:
            fit_info += '    ' + 'id= ' + str(id)

        fig.suptitle(suptitle+'\n'+fit_info, fontdict={'fontsize': 10})


        if plot_settings.get('plot_guess', None) is not None:
            ax1.plot(self._init_guess_y.real, self._init_guess_y.imag, '--', label='inital fit', c='#d1d1e0')
            ax2.plot(self._datax, 20*np.log10(np.abs(self._init_guess_y)), '--', label='inital fit', c='#d1d1e0')

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
        for param in self.result.params:
            stderr[param] = self.result.params[param].stderr
        return stderr

    def _init_params(self):
        return self._params.valuesdict()

    def _params_name(self):
        return list(self._params.valuesdict().keys())



def params(name: str):
    if name in ['ComplexResonatorModel', 'ResonatorModel']:
        _params = getattr(md, name)().param_names
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
    mag = df['S21 magnitude'].to_numpy()[_power_mask]
    phase = df['S21 phase'].to_numpy()[_power_mask]
    S21 = mag * np.exp(1j * phase)

    # Scale the frequency
    freq = freq * 1e-9

    return freq, S21

def resonator_fit_all(file_location: str, power_limit=None):
    import pandas as pd
    df = pd.read_csv(file_location, delimiter='\t', header=0,
                     skiprows=lambda x: x in [0, 2])
    df.columns = ['Power'] + list(df.columns[1:])
    power = df['Power'].unique()
    Qi_list, Qi_err = [], []
    Qe_list, Qe_err = [], []
    if power_limit:
        power = power[(power > power_limit[0]) & (power < power_limit[1])]
    for p in power:
        freq, S21 = read_dat(file_location, power=p)
        t3 = QFit(freq, S21, model='ResonatorModel')
        t3.guess()
        t3.do_fit()
        qierr = t3.err_params('Qi')
        qeerr = t3.err_params('Qe_mag')
        if t3.err_params('Qi') is None:
            for _win in range(2, 7):
                freq, S21 = read_dat(file_location, power=p)
                t3 = QFit(freq, S21, model='ResonatorModel')
                t3.guess()
                t3.wash(method='complexcomp', window=_win*1e-2)
                t3.do_fit()
                qierr = t3.err_params('Qi')
                qeerr = t3.err_params('Qe_mag')
                if t3.err_params('Qi') is not None:
                    break
        if t3.err_params('Qi') is None:
            qierr = 0.2*t3.fit_params('Qi')
            qeerr = 0.2*t3.fit_params('Qe_mag')
        Qi_list.append(t3.fit_params('Qi')*1e3)
        Qi_err.append(qierr*1e3)
        Qe_list.append(t3.fit_params('Qe_mag')*1e3)
        Qe_err.append(qeerr*1e3)
    fig, ax = plt.subplots()
    ax.errorbar(power, Qi_list, yerr=Qi_err, fmt='o', c='r', label='Qi')
    ax.set_xlabel('Power(dB)')
    ax.set_ylabel('$Q_{int}$', fontsize=14)
    ax2 = ax.twinx()
    ax2.errorbar(power, Qe_list, yerr=Qe_err, fmt='x', c='c', label='Qe')
    ax2.set_ylabel('$Q_{ext}$', fontsize=14)
    ax.legend(loc="lower left")
    ax.set_ylim(top=1.1*max(Qi_list), bottom=0.9*min(Qi_list))
    ax2.set_ylim(top=1.1*max(Qe_list), bottom=0.9*min(Qe_list))
    plt.tight_layout()
    plt.show()


