from lmfit import Model, Minimizer, Parameters, report_fit, models
import numpy as np
import matplotlib.pyplot as plt




class QFit:
    def __init__(self, data_x=None, data_y=None, model=None, params_init=None, method='leastsq'):
        self._datax = data_x.flatten()
        self._datay, self._fity = data_y.flatten(), np.empty((0, len(data_y)))
        self.result = 0
        self.method = method

        # define the history of y, use for pdf or plots

        if isinstance(model, str):
            result = getattr(models, model)()
        else:
            self._qmodel = Model(model)

        self._params = self._qmodel.make_params()
        if params_init is not None:
            for n_params in range(len(params_init)):
                print(n_params)
                self._params.add(self._qmodel.param_names[n_params], params_init[n_params])

    def set_params(self, name: str, value: float=None, vary: bool=True, minimum=None, maximum=None, expression=None
                   , brute_step=None):
        self._params.add(name, value, vary, minimum, maximum, expression, brute_step)


    def do_fit(self):
        self.result = self._qmodel.fit(self._datay, self._params, x=self._datax, method=self.method)
        print(self.result.fit_report())
        self._fity = np.vstack((self._fity, self.result.best_fit.T))

    def plot_show(self):
        try:
            import matplotlib.pyplot as plt
            for row in range(len(self._fity)):
                plt.plot(self._datax, self._datay, 'k+')
                plt.plot(self._datax, self._fity[row], 'r')
        except ImportError:
            pass

    def params_stderr(self):
        stderr = {}
        for params in self.result.params.keys():
            stderr[params] = self.result.params.get(params).stderr
        return stderr


    def pretty_print(self, plot_settings=None):
        '''Basic function for plotting the result of a fit'''
        fit_params, error_params = self.result.best_values, self.params_stderr()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Add the original data
        data_color = 'C0' if plot_settings is None else plot_settings.get('data_color', 'C0')
        ax.plot(self._datax, self._fity.flatten(), '.', label='Data', color=data_color, markersize=10, zorder=10)
        fit_color = 'gray' if plot_settings is None else plot_settings.get('fit_color', 'k')
        # Add fitting curve:
        ax.plot(self._datax, self._fity.flatten(), '-', linewidth=1, label='Fit', color=fit_color)
        ax.plot(self._datax, self._fity.flatten(), 'o', markersize=3, color=fit_color)
        # Hack to add legend with fit-params:
        for key in fit_params.keys():
            ax.plot(self._datax[0], self._fity.flatten()[0], 'o', markersize=0, \
                    label='{​}​: {​:4.4}​±{​:4.4}​'.format(key, fit_params[key], error_params[key]))
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
        ax.set_title('Datasource: ' + title + ' Fit type: ' + str(self.FitFunction.__name__))
        plt.show()
        # Check if use wants figure and return if needed:
        if plot_settings.get('return_fig') is not None:
            return fig, ax


    def pdf_print(self):
        import datetime
        from matplotlib.backends.backend_pdf import PdfPages

        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.
        with PdfPages('multipage_pdf.pdf') as pdf:
            plt.figure(figsize=(3, 3))
            plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
            plt.title('Page One')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

            # if LaTeX is not installed or error caught, change to `False`
            plt.rcParams['text.usetex'] = True
            plt.figure(figsize=(8, 6))
            x = np.arange(0, 5, 0.1)
            plt.plot(x, np.sin(x), 'b-')
            plt.title('Page Two')
            pdf.attach_note("plot of sin(x)")  # attach metadata (as pdf note) to page
            pdf.savefig()
            plt.close()

            plt.rcParams['text.usetex'] = False
            fig = plt.figure(figsize=(4, 5))
            plt.plot(x, x ** 2, 'ko')
            plt.title('Page Three')
            pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
            plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Multipage PDF Example'
            d['Author'] = 'Jouni K. Sepp\xe4nen'
            d['Subject'] = 'How to create a multipage pdf file and set its metadata'
            d['Keywords'] = 'PdfPages multipage keywords author title subject'
            d['CreationDate'] = datetime.datetime(2009, 11, 13)
            d['ModDate'] = datetime.datetime.today()
        pass

