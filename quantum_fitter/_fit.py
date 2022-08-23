from lmfit import Model, Minimizer, Parameters, report_fit, models
import numpy as np
import matplotlib.pyplot as plt
import os
import quantum_fitter._model as md


class QFit:
    def __init__(self, data_x, data_y=None, model=None, params_init=None, method="least_squares", **kwargs):
        self._raw_y = data_y.flatten()
        # data_y /= np.mean(np.abs(data_y)[[0, -1]])
        self._datax = data_x.flatten()
        self._fitx = None
        self._datay, self._fity = data_y, None
        # define the history of y, use for pdf or plots
        if self._datay is not None:
            self._datay, self._fity = data_y.flatten(), np.empty((0, len(data_y)))
        self.result = 0
        self.method = method
        self._fig, self._ax = 0, 0
        self.wash_status = False
        self._init_guess_y = None
        self._init_guess_params = None

        self.makemodels(model, params_init)

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

    def makemodels(self, model, params_init) -> None:
        # using default build-in model in lmfit. If want to do multiple build-in model, just pass in a list of str
        # Example: model=['LinearModel', 'LorentzianModel']
        if isinstance(model, list):
            self._qmodel = self.makemodel(model[0])

            if len(model) > 1:
                for i, m in enumerate(model[1:]):
                    mod = self.makemodel(m)
                    print(m)
                    print(model[: i + 1])
                    if m in model[: i + 1]:
                        mod.prefix = f"f{i+2}_"
                    self._qmodel += mod
        else:
            self._qmodel = self.makemodel(model)

        self._params = self._qmodel.make_params()

    def makemodel(self, model) -> Model:
        if isinstance(model, str):
            return self.strtomodel(model)
        elif isinstance(model, Model):
            return model
        elif callable(model):
            return Model(model)

    def strtomodel(self, model):
        if model in ["ComplexResonatorModel", "ResonatorModel"]:
            return getattr(md, model)()
        else:
            return getattr(models, model)()

    def __str__(self):
        return "Lmfit hi"

    def set_params(
        self,
        name: str,
        value: float = None,
        vary: bool = True,
        minimum=None,
        maximum=None,
        expression=None,
        brute_step=None,
    ):
        self._params.add(name, value, vary, minimum, maximum, expression, brute_step)

    @property
    def params(self):
        """
        Set or get the parameters of current models.
        :return: Parameters' dictionary
        """
        print(self._params.valuesdict())
        return self._params.valuesdict()

    @params.setter
    def params(self, init_dict: dict):
        for para_name in init_dict.keys():
            self._params.add(para_name, init_dict[para_name])

    @property
    def data_y(self):
        if not self._datay:
            print("No data y now!")
            return
        return self._datay

    @data_y.setter
    def data_y(self, data_y):
        self._datay = data_y

    @property
    def fit_y(self):
        if len(self._fity) == 0:
            print("No data y now!")
            return
        return self._fity

    @fit_y.setter
    def fit_y(self, fit_y):
        print("here")
        self._fity = fit_y

    def make_params(self, **kwargs):
        return self._qmodel.make_params(**kwargs)

    def add_models(self, model, merge: str = "+"):
        """
        Add model to current models.
        :param model: The model you want to add in.
        :param merge: The operation needed to merge, can be +,-,/,*.
        :return:
        """
        if isinstance(model, str):
            _new_model = getattr(models, model)()
        else:
            _new_model = Model(model)
        # Check if there is any same parameter name
        name_list = set(self._qmodel.param_names).intersection(_new_model.param_names)
        if name_list:
            print("The build-in models have the same parameter name" + str(name_list))
            if isinstance(model, str):
                model_name = model
                prefix = "".join([c for c in model if c.isupper()])
            else:
                prefix = model.__name__
                model_name = prefix
            _new_model.prefix = prefix
            print("Add prefix", prefix, "to the parameters in", model_name)

        if merge == "+":
            self._qmodel += _new_model
        elif merge == "*":
            self._qmodel *= _new_model
        elif merge == "-":
            self._qmodel -= _new_model
        elif merge == "/":
            self._qmodel /= _new_model
        else:
            self._qmodel += _new_model
            print("Merge style wrongly specified. Using '+' operator instead\n")
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
        self.wash(method="savgol", window_length=level, polyorder=3)

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
            self.weight = np.exp(-((weight_x - muu) ** 2 / (2.0 * _sigma**2)))
            self.weight = (self.weight + 0.1) / 1.1

    def do_fit(self, report=None):
        self._init_guess_params = self._params.copy()

        self.result = self._qmodel.fit(
            self._datay,
            self._params,
            x=self._datax,
            method=self.method,
            weights=self.weight,
            nan_policy="omit",
        )
        self._params = self.result.params
        if report:
            print(self.result.fit_report())
        self._fity = self.result.best_fit
        if self.wash_params:
            self._fity = self._fity * np.exp(1j * self.wash_params[0] * self._datax)

    def wash(self, method="savgol", **kwargs):
        if method == "savgol":
            from scipy.signal import savgol_filter

            _win_l = kwargs.get("window_length") if kwargs.get("window_length") else 3
            _po = kwargs.get("polyorder") if kwargs.get("polyorder") else 2
            if np.iscomplexobj(self._datay):
                rr = savgol_filter(np.real(self._datay), _win_l, _po)
                ri = savgol_filter(np.imag(self._datay), _win_l, _po)
                self._datay = np.vectorize(complex)(rr, ri)
            else:
                self._datay = savgol_filter(self._datay, _win_l, _po)

        if method == "cut":
            _win = kwargs.get("window") if kwargs.get("window") else [1 / 3, 2 / 3]
            self._datax = self._datax[int(len(self._datax) * _win[0]) : int(len(self._datax) * _win[1])]
            self._datay = self._datay[int(len(self._datay) * _win[0]) : int(len(self._datay) * _win[1])]
            self._fity = self._fity[int(len(self._fity) * _win[0]) : int(len(self._fity) * _win[1])]
            self._raw_y = self._raw_y[int(len(self._raw_y) * _win[0]) : int(len(self._raw_y) * _win[1])]

        if method == "linearcomp":
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

        if method == "complexcomp":
            # inspired and from David's code
            _window = (
                int(kwargs.get("window") * len(self._datax)) if kwargs.get("window") else int(0.06 * len(self._datax))
            )
            phase = np.unwrap(np.angle(self._datay))
            line_fit = np.polyfit(
                [np.mean(self._datax[:_window]), np.mean(self._datax[-_window:])],
                [np.mean(phase[:_window]), np.mean(phase[-_window:])],
                1,
            )

            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax).real, label='lcreal')
            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax).imag, label='lcimag')
            # plt.plot(self._datax, self._datay.imag)
            self._datay = self._datay * np.exp(-1j * line_fit[0] * self._datax)
            # plt.plot(self._datax, np.exp(-1j * line_fit[0] * self._datax))
            # plt.plot(self._datax, self._datay.imag, ls='--', c='grey')
            self.wash_params = [line_fit[0]]

        if method == "fft":
            from scipy.fft import rfft, rfftfreq, irfft
            from scipy.signal import savgol_filter

            if np.iscomplexobj(self._datay):
                yf = rfft(self._datay.real)
                xf = rfftfreq(len(self._datay.real), 1 / len(self._datay.real))
                yfi = rfft(self._datay.imag)
                xfi = rfftfreq(len(self._datay.imag), 1 / len(self._datay.imag))

                target_idx = int(len(xf) / 64)
                yf[target_idx:] = savgol_filter(yf[target_idx:], window_length=5, polyorder=3)
                # yf[target_idx:] = 0
                self._datay = irfft(yf).astype("complex128")

                target_idx = int(len(xfi) / 64)
                yfi[target_idx:] = savgol_filter(yfi[target_idx:], window_length=5, polyorder=3)
                # yfi[target_idx:] = 0
                self._datay += 1j * irfft(yfi).astype("complex128")
                self._datay = np.append(self._datay, self._datay[-1])
                self._datay[:8] = self._datay[8:16]
                self._datay[-8:] = self._datay[-16:-8]

            if method == "focus":
                np.argmin(np.log10(np.abs(self._datay)))

    def cov_mat(self):
        return self.result.covar

    def complex_comp(self):
        _window = len(self._datax)
        phase = np.unwrap(np.angle(self._raw_y))
        line_fit = np.polyfit(
            [np.mean(self._datax[:_window]), np.mean(self._datax[-_window:])],
            [np.mean(phase[:_window]), np.mean(phase[-_window:])],
            1,
        )
        return line_fit[0]

    def plot_cov_mat(self):
        f = plt.figure(figsize=(8, 6))
        _cov_mat = self.cov_mat()
        _tick = self._params_name()
        print(_cov_mat)
        plt.matshow(_cov_mat, fignum=f.number)
        plt.xticks(range(len(_tick)), _tick, fontsize=10, rotation=45)
        plt.yticks(range(len(_tick)), _tick, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)
        plt.title("Correlation Matrix", fontsize=11)

    def init_eval(self, xdata):
        return self.eval(self._init_guess_params, x=xdata)

    def pretty_print(self, plot_settings=None, x=None):
        """Basic function for plotting the result of a fit"""
        if x is not None:
            if isinstance(x, int):
                self._fitx = np.linspace(min(self._datax), max(self._datax), 100)
            else:
                self._fitx = x
            fit_value = self.eval(x=self._fitx)
        else:
            fit_value = self.fit_y
            self._fitx = self._datax
        fit_params, error_params = self.result.best_values, self._params_stderr()
        _fig_size = (8, 6) if plot_settings is None else plot_settings.get("fig_size", (8, 6))
        self._fig, ax = plt.subplots(1, 1, figsize=_fig_size)
        # Add the original data
        data_color = "C0" if plot_settings is None else plot_settings.get("data_color", "C0")
        ax.plot(self._datax, self._datay, ".", label="Data", color=data_color, markersize=10, zorder=10)

        if plot_settings.get("plot_guess", None) is not None:
            ax.plot(self._fitx, self.init_eval(self._fitx), "--", label="inital fit", c="#d1d1e0")

        fit_color = "gray" if plot_settings is None else plot_settings.get("fit_color", "k")
        # Add fitting curve:
        ax.plot(self._fitx, fit_value, "-", linewidth=1, label="Fit", color=fit_color)
        if x is None:
            ax.plot(self._fitx, fit_value, "o", markersize=3, color=fit_color)
        # Hack to add legend with fit-params:
        for key in fit_params.keys():
            ax.plot(self._fitx[0], fit_value[0], 'o', markersize=0,
                    label='{}: {:4.4f}±{:4.4f}'.format(key, fit_params[key], str_none_if_none(error_params[key])))
        # Rescale plot if user wants it:
        if plot_settings is not None:
            ax.set_xlabel(plot_settings.get("x_label", "x_label not set"))
            ax.set_ylabel(plot_settings.get("y_label", "y_label not set"))
            if "x_lim" in plot_settings.keys():
                ax.set_xlim(plot_settings["x_lim"])
            if "y_lim" in plot_settings.keys():
                ax.set_ylim(plot_settings["y_lim"])

        ax.legend()
        title = "Data source not given" if plot_settings is None else plot_settings.get("plot_title", "no name given")
        fit_type = (
            str(self._qmodel.name) if plot_settings is None else plot_settings.get("fit_type", str(self._qmodel.name))
        )
        ax.set_title("Datasource: " + title + "\n Fit type: " + fit_type)
        # Check if use wants figure and return if needed:
        if plot_settings is None or plot_settings.get("show_fig", None) is None:
            plt.show()

        if plot_settings is not None and plot_settings.get("return_fig", None) is not None:
            return self._fig, ax

    def polar_plot(self, plot_settings={}, power=99999, f0=None, id=None, suptitle=""):
        angle = np.exp(-1j * (self._datax * self.result.params["phi1"] + self.result.params["phi2"]))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 3))
        ax1.plot(self._fity.real, self._fity.imag, "r", label="best fit", linewidth=1.5)
        ax1.scatter(self._raw_y.real, self._raw_y.imag, c="grey", s=1)
        ax1.set_title("Raw S21 Complex Plane", fontdict={"size": 10})
        ax1.set_xlabel("Re(S21)")
        ax1.set_ylabel("Im(S21)")

        ax2.plot((self._fity * angle).real, (self._fity * angle).imag, "r", label="best fit", linewidth=1.5)
        ax2.scatter((self._raw_y * angle).real, (self._raw_y * angle).imag, c="grey", s=1)
        ax2.set_title("Raw S21 Complex Plane", fontdict={"fontsize": 10})
        ax2.set_xlabel("Re(S21)")
        ax2.set_ylabel("Im(S21)")

        ax3.plot(self._datax, 20 * np.log10(np.abs(self._fity)), "r", label="best fit", linewidth=1.5)
        ax3.scatter(self._datax, 20 * np.log10(np.abs(self._raw_y)), c="grey", s=1)
        ax3.scatter(
            self._datax[np.argmin(20 * np.log10(np.abs(self._raw_y)))],
            np.min(20 * np.log10(np.abs(self._raw_y))),
            c="b",
            s=5,
        )
        ax3.set_title("S21 Mag", fontdict={"fontsize": 10})
        ax3.set_xlabel("Frequency / GHz")
        ax3.set_ylabel("S21(dB)")
        ax3.ticklabel_format(useOffset=False)

        ax4.plot(self._datax, np.angle(self._fity * angle), "r", label="best fit", linewidth=1.5)
        ax4.scatter(self._datax, np.angle(self._raw_y * angle), c="grey", s=1)
        ax4.set_title("S21 Phase", fontdict={"fontsize": 10})
        ax4.set_xlabel("Frequency / GHz")
        ax4.set_ylabel("Angle / rad")
        ax4.ticklabel_format(useOffset=False)

        # if self._qmodel.name == 'ResonatorModel':
        fit_info = "$Q_{int}= $" + str("{0:.1f}".format(self.fit_params("Qi") * 1e3)) + "    "
        fit_info += "$Q_{ext}= $" + str("{0:.1f}".format(self.fit_params("Qe_mag") * 1e3))
        # if self._qmodel.name == 'ResonatorModel':
        #     Qe = self.fit_params('Q_e_real') + 1j * self.fit_params('Q_e_imag')
        #     Qi = 1 / (1 / self.fit_params('Q') - 1 / self.fit_params('Q_e_real')) * 1e3
        #     fit_info = '$Q_{int}= $' + str("{0:.1f}".format(self.fit_params('Qi') * 1e3)) + '    '
        #     fit_info += '$Q_{ext}= $' + str("{0:.1f}".format(self.fit_params('Qe_mag') * 1e3))

        if power != 99999:
            fit_info += "    " + "$P_{VNA}=$ " + str(power) + "dBm"
        if f0:
            fit_info += "    " + "f0= " + str("{0:.4f}".format(f0)) + "GHZ"
        if id:
            fit_info += "    " + "id= " + str(id)

        fig.suptitle(suptitle + "\n" + fit_info, fontdict={"fontsize": 10})

        if plot_settings.get("plot_guess", None) is not None:
            ax1.plot(self._init_guess_y.real, self._init_guess_y.imag, "--", label="inital fit", c="#d1d1e0")
            ax3.plot(self._datax, 20 * np.log10(np.abs(self._init_guess_y)), "--", label="inital fit", c="#d1d1e0")

        fig.tight_layout()

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
            d["Title"] = "Qfit PDF Example"
            d["Author"] = "Kian"
            d["Subject"] = "Qfit"
            d["CreationDate"] = datetime.datetime.today()
            d["ModDate"] = datetime.datetime.today()
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
    if name in ["ComplexResonatorModel", "ResonatorModel"]:
        _params = getattr(md, name)().param_names
        print(name + "'s parameters: " + str(_params))
        return _params
    _params = getattr(models, name)().param_names
    print(name + "'s parameters: " + str(_params))
    return _params


def read_dat(file_location: str, power):
    import pandas as pd

    df = pd.read_csv(file_location, delimiter="\t", header=0, skiprows=lambda x: x in [0, 2])
    df.columns = ["Power"] + list(df.columns[1:])
    _power = df["Power"].to_numpy()
    _power_mask = np.argwhere(_power == power)  # Choose the power you fit
    freq = df["S21 frequency"].to_numpy()[_power_mask]
    mag = df["S21 magnitude"].to_numpy()[_power_mask]
    phase = df["S21 phase"].to_numpy()[_power_mask]
    S21 = mag * np.exp(1j * phase)

    # Scale the frequency
    freq = freq * 1e-9

    return freq, S21


def resonator_fit_all(file_location: str, power_limit=None):
    import pandas as pd

    df = pd.read_csv(file_location, delimiter="\t", header=0, skiprows=lambda x: x in [0, 2])
    df.columns = ["Power"] + list(df.columns[1:])
    power = df["Power"].unique()
    Qi_list, Qi_err = [], []
    Qe_list, Qe_err = [], []
    if power_limit:
        power = power[(power > power_limit[0]) & (power < power_limit[1])]
    for p in power:
        _success = False
        for _win in range(0, 2, 7):
            print(p)
            freq, S21 = read_dat(file_location, power=p)
            t3 = QFit(freq, S21, model="ResonatorModel")
            t3.guess()
            if _win != 0:
                t3.wash(method="complexcomp", window=_win * 1e-2)
            # t3.wash(method='fft')
            t3.do_fit()
            qierr = t3.err_params("Qi")
            print(qierr)
            qeerr = t3.err_params("Qe_mag")

            # Check if fit fails?
            if t3.err_params("Qi") is not None:
                if t3.fit_params("Qi") < 1e5:
                    if t3.err_params("Qi") < 0.5 * t3.fit_params("Qi"):
                        Qi_list.append(t3.fit_params("Qi") * 1e3)
                        Qi_err.append(qierr * 1e3)
                        Qe_list.append(t3.fit_params("Qe_mag") * 1e3)
                        Qe_err.append(qeerr * 1e3)
                        _success = True
                        break

        if _success is False:
            print("Not able to estimate in this power: " + str(p))
            Qi_list.append(0)
            Qi_err.append(0)
            Qe_list.append(0)
            Qe_err.append(0)

    fig, ax = plt.subplots()
    print(power, Qi_list)
    ax.errorbar(power, Qi_list, yerr=Qi_err, fmt="o", c="r", label="Qi")
    ax.set_xlabel("Power(dB)")
    ax.set_ylabel("$Q_{int}$", fontsize=14, c="r")
    ax2 = ax.twinx()
    ax2.errorbar(power, Qe_list, yerr=Qe_err, fmt="x", c="c", label="Qe")
    ax2.set_ylabel("$Q_{ext}$", fontsize=14, c="c")
    ax.legend(loc="lower left")
    ax.set_ylim(top=1.1 * max(Qi_list) if max(Qi_list) < 1e7 else 1e6, bottom=0.9 * min(Qi_list))
    ax2.set_ylim(top=1.1 * max(Qe_list) if max(Qe_list) < 1e7 else 1e6, bottom=0.9 * min(Qe_list))
    plt.tight_layout()
    plt.show()


def str_none_if_none(stderr):
    if stderr is None:
        return "None"
    else:
        return stderr


def oddfun_damped_oscillations_guess(x, y):
    # Adapted from QDev wrappers, `qdev_fitter`
    from scipy import fftpack

    a = (y.max() - y.min()) / 2
    c = y.mean()
    T = x[round(len(x) / 2)]
    yhat = fftpack.rfft(y - y.mean())
    idx = (yhat**2).argmax()
    freqs = fftpack.rfftfreq(len(x), d=(x[1] - x[0]) / (2 * np.pi))
    w = freqs[idx]

    dx = x[1] - x[0]
    indices_per_period = np.pi * 2 / w / dx
    std_window = round(indices_per_period)
    initial_std = np.std(y[:std_window])
    noise_level = np.std(y[-2 * std_window :])
    for i in range(1, len(x) - std_window):
        std = np.std(y[i : i + std_window])
        if std < (initial_std - noise_level) * np.exp(-1):
            T = x[i]
            break
    p = 0
    return [a, T, w, p, c]


def oddfun_damped_oscillations(x, A, T, omega, phi, c):
    # E.g. for fitting Rabis
    return A * np.sin(omega * x + phi) * np.exp(-x / T) + c


def oddfun_oscillations(x, A, omega, phi, c):
    # E.g. for fitting Rabis
    return A * np.sin(omega * x + phi) + c


def exp_func(x, A, T, c):
    return A * np.exp(-x / T) + c


def weighted_mean(data, errors):
    from scipy import stats
    import numpy.ma as ma
    import numpy as np

    """Calculates weighted mean of data areay with errors.

    Args:
        data (array): Array of data, shape = [N,1]
        errors (array): Array of errors for data, shape = [N,1]

    Returns:
        mean_weighted (float): Number of the weighted mean
        err_weighted (float): Number of the weighted error
        Chi2 (float): Chi2 of the arrays
        Chi2_prob (float): Chi2 Probability of the array
    """

    data = np.array(data)
    errors = np.array(errors)
    data = data[errors != None]
    errors = errors[errors != None]

    weight = 0
    for i in errors:
        if i == 0:
            weight = 1
        else:
            if i == None:
                weight += 0
            else:
                weight += (1 / i) ** 2

    mean_weighted = 0
    for i in range(len(data)):
        try:
            mean_weighted += (float(data[i]) * (1 / float(errors[i])) ** 2) / weight
        except:
            pass

    err_weighted = np.sqrt(1 / weight)

    Chi2 = np.sum(((np.array(data) - mean_weighted) / np.array(errors)) ** 2)
    Chi2_prob = stats.chi2.sf(Chi2, len(data) - 1)

    return mean_weighted, err_weighted, Chi2, Chi2_prob


def avg_plot(path, mode="T1", figsize=(8, 6), mask=False):
    """This function takes the average of entries in the dataset and plots them.

    Args:
        path (string)): datafile path as a string
        mode (str, optional): The fitting mode. Can also be 'T2'. Defaults to 'T1'.
        figsize (tuple, optional): Size of the plt. Defaults to (8, 6).
        mask (bool, optional): The range of datapoints to remove. If False all datapoints are used. Defaults to False.
    """
    import Labber as lab
    import quantum_fitter as qf
    import quantum_fitter.standard_operating_functions as sof

    file = lab.LogFile(path)
    filename = qf.get_file_name_from_path(path)

    X, y_arr = file.getTraceXY(entry=0)
    for i in range(file.getNumberOfEntries() - 1):
        _, yi = file.getTraceXY(entry=i + 1)
        y_arr = np.vstack([y_arr, yi])

    y_avg = np.mean(y_arr, axis=0)

    X *= 1e6
    y_avg *= 1e6

    angle = sof.calcRotationAngle(y_avg)  # find angle in radians to rotate data by
    y = np.real(np.exp(1j * angle) * y_avg)

    if mask != False:
        mask_ = (X < mask[0]) | (X > mask[1])
        X = X[mask_]
        y = y[mask_]

    if mode == "T1":
        fit_type = r"$A \times exp(-x/T) + c$"

        t2 = qf.QFit(X, y, model=Model(exp_func))
        t2.set_params("T", 100)
        t2.set_params("A", 100)
        t2.set_params("c", 100)

    if mode == "T2":
        fit_type = r"$A \times exp(-x/T) \times sin(\omega x + \varphi) + c$"

        a, T, w, p, c = oddfun_damped_oscillations_guess(X, y)

        # fitting
        t2 = qf.QFit(X, y, model=Model(oddfun_damped_oscillations))
        t2.set_params("T", T)
        t2.set_params("A", a)
        t2.set_params("c", c)
        t2.set_params("omega", w)
        t2.set_params("phi", p)

    t2.do_fit()

    t2.pretty_print(
        plot_settings={
            "x_label": "Sequence duration (\u03BCs)",
            "y_label": r"$V_{H}$" " (\u03BCV)",
            "plot_title": f"{filename}, Averaged",
            "fit_type": fit_type,
            "fit_color": "C4",
            "fig_size": figsize,
        },
        x=0,
    )


def multi_entry(
    path,
    plot_i=[],
    mode="T1",
    plot_mean=True,
    time_axis=True,
    figsize=(8, 6),
    mask=False,
    entry_mask=[],
    plot_hist=False,
    return_object=False,
    return_freq=False,
    fit_guess={},
    verbose=False,
    use_phase=False,
):
    """Take a dataset and cal the decay (T1 or T2). Returns a plot of the decays.

    Args:
        path (str): The datafile path
        plot_i (list, optional): The number of subplots you want to plot. Defaults to [].
        mode (str, optional): Can be 'T1' or 'T2'. Defaults to 'T1'.
        plot_mean (bool, optional): Plots mean and error, if True. Defaults to True.
        time_axis (_type_, optional): Plots one a timescale if True. Defaults to None.
        figsize (tuple, optional): The size of the plot. Defaults to (8, 6).
        mask (bool, optional): The range of datapoints to remove. If False all datapoints are used. Defaults to False.
    """
    import quantum_fitter as qf
    import Labber as lab
    import quantum_fitter.standard_operating_functions as sof

    file = lab.LogFile(path)
    filename = qf.get_file_name_from_path(path)

    if plot_i == False:
        plot_i = []

    rep, t2_array, t2_error, time_array, omega_array, omega_error = [], [], [], [], [], []
    entry_0 = file.getEntry(entry=0)["timestamp"] / 60

    for i in range(file.getNumberOfEntries()):
        if i not in entry_mask:
            X, y = file.getTraceXY(entry=i)

            entry = file.getEntry(entry=i)
            time_i = entry["timestamp"] / 60 - entry_0
            time_array.append(time_i)

            # rescaling units
            X *= 1e6
            y *= 1e6

            if use_phase:
                y = np.angle(y)
            else:
                angle = sof.calcRotationAngle(y)  # find angle in radians to rotate data by
                y = np.real(np.exp(1j * angle) * y)

            if mask != False:
                mask_ = (X < mask[0]) | (X > mask[1])
                X = X[mask_]
                y = y[mask_]

            if mode == "T1":
                fit_type = r"$A \times exp(-x/T) + c}$"

                baseline = np.mean(y[-int(0.1 * len(y)) : -1])
                top = np.mean(y[0 : int(0.1 * len(y))])
                t1_est = X[np.abs(y - top / 2).argmin()]

                print(baseline)
                print(top - baseline)
                print(t1_est)

                t2 = qf.QFit(X, y, model=Model(exp_func))
                t2.set_params("T", t1_est)
                t2.set_params("A", top - baseline)
                t2.set_params("c", baseline)

            if mode == "T2":
                fit_type = r"$A \times exp(-x/T) \times sin(\omega x + \varphi) + c$"

                a, T, w, p, c = oddfun_damped_oscillations_guess(X, y)
                print(a, T, w, p, c)

                # fitting
                t2 = qf.QFit(X, y, model=Model(oddfun_damped_oscillations))
                t2.set_params("T", T)
                t2.set_params("A", a)
                t2.set_params("c", c)
                t2.set_params("omega", w)
                t2.set_params("phi", p)
                for param, value in fit_guess.items():
                    t2.set_params(param, value)

            t2.do_fit()

            # plotting i entry
            if plot_i == True or i in plot_i:
                t2.pretty_print(
                    plot_settings={
                        "x_label": "Sequence duration (\u03BCs)",
                        "y_label": r"$V_{H}$" " (\u03BCV)",
                        "plot_title": f"{filename}, Repetition: {i}",
                        "fit_type": fit_type,
                        "fit_color": "C4",
                        "fig_size": figsize,
                    },
                    x=0,
                )

            t2_error.append(t2.err_params("T"))
            t2_array.append(t2.fit_params("T"))
            rep.append(i + 1)
            if mode == "T2":
                omega_array.append(t2.fit_params("omega"))
                omega_error.append(t2.err_params("omega"))

    # print(mean,'\u00B1', error)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Repetitions"), ax.set_ylabel(f"{mode} decay (\u03BCs)")

    # plot final
    if time_axis != None:
        rep = time_array
        ax.set_xlabel("Time (min)")

    t2_array = np.array(t2_array)
    t2_error = np.array(t2_error)
    rep = np.array(rep)

    good_indices = t2_error != None
    rep_good = rep[good_indices]
    t2_array_good = t2_array[good_indices]
    t2_error_good = t2_error[good_indices]

    good_indices = abs(t2_error_good / t2_array_good) < 1
    rep_good = rep_good[good_indices]
    t2_array_good = t2_array_good[good_indices]
    t2_error_good = t2_error_good[good_indices]

    plt.errorbar(rep_good, t2_array_good, t2_error_good, fmt=".", color="red", ecolor="grey")

    if plot_mean:
        mean, error, chi2, chi2_prob = weighted_mean(t2_array, t2_error)
        plt.axhline(y=mean, color="r", linestyle="-", label=f"weighted mean: {mean:.3} \u00B1 {error:.2}")
        plt.axhline(y=mean + error, color="grey", linestyle="--")
        plt.axhline(y=mean - error, color="grey", linestyle="--")
        plt.ylim([0, np.max(mean) * 2])

    plt.legend()
    plt.title(f"{filename}\nFit type: {fit_type}")
    # plt.tight_layout()
    plt.show()

    if plot_hist:
        bins = int(np.sqrt(np.sum(t2_array)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(f"{mode} decay (\u03BCs) [{bins} bins]"), ax.set_ylabel("Counts")

        ax.hist(t2_array, bins=bins)

        plt.title(f"{filename}\nFit type: {fit_type}")
        plt.show()

    if return_freq:
        return omega_array, omega_error

    if return_object:
        return t2
