from lmfit import Model, Minimizer, Parameters, report_fit, models
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import os

class QFit:
    def __init__(self, data_x, data_y, model=None, params_init=None, method='leastsq'):
        self._datax = data_x.flatten()

        # define the history of y, use for pdf or plots
        self._datay, self._fity = data_y.flatten(), np.empty((0, len(data_y)))
        self.result = 0
        self.method = method
        self._fig, self._ax = 0, 0
        self.wash_status = False
        self._origin_data = self._datay

        # using default build-in model in lmfit. If want to do multiple build-in model, just pass in a list of str
        # Example: model=['LinearModel', 'LorentzianModel']
        if isinstance(model, str):
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

    def __str__(self):
        return 'Lmfit hi'

    def set_params(self, name: str, value: float = None, vary: bool = True, minimum=None, maximum=None, expression=None
                   , brute_step=None):
        self._params.add(name, value, vary, minimum, maximum, expression, brute_step)

    @property
    def params(self):
        print(self._params.valuesdict())
        return self._params.valuesdict()

    @params.setter
    def params(self, init_dict: dict):
        for para_name in init_dict.keys():
            self._params.add(para_name, init_dict[para_name])

    def add_models(self, model, merge: str = '+'):
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

    def fit_params(self, name: str = None):
        if name is None:
            return self.result.values
        return self.result.values[name]

    def err_params(self, name: str = None):
        if name is None:
            return self._params_stderr()
        return self._params_stderr()[name]

    def fit_values(self):
        return self.result.best_fit

    def do_fit(self):
        self.result = self._qmodel.fit(self._datay, self._params, x=self._datax, method=self.method)
        print(self.result.fit_report())
        self._fity = np.vstack((self._fity, self.result.best_fit.T))

    def wash(self, method='savgol', **kwargs):
        print('Using method '+method+' to smooth the data')
        if method == 'savgol':
            self._datay = savgol_filter(self._datay, kwargs.get('window_length'), kwargs.get('polyorder'))

    def plot_show(self):
        try:
            import matplotlib.pyplot as plt
            for row in range(len(self._fity)):
                plt.plot(self._datax, self._datay, 'k+')
                plt.plot(self._datax, self._fity[row], 'r')
        except ImportError:
            pass

    def pretty_print(self, plot_settings=None):
        '''Basic function for plotting the result of a fit'''
        fit_params, error_params, fit_value = self.result.best_values, self._params_stderr(), \
                                              self.result.best_fit.flatten()
        _fig_size = (8, 6) if plot_settings is None else plot_settings.get('fig_size', (8, 6))
        fig, ax = plt.subplots(1, 1, figsize=_fig_size)
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

        try:
            if plot_settings.get('return_fig', None) is not None:
                return fig, ax
        except:
            pass

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

    def _params_stderr(self):
        stderr = {}
        for params in self.result.params.keys():
            stderr[params] = self.result.params.get(params).stderr
        return stderr

    def _init_params(self):
        return self._params.valuesdict()


def params(name: str):
    _params = getattr(models, name)().param_names
    print(name + '\'s parameters: ' + str(_params))
    return _params



