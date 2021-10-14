import sys
import quantum_fitter as qf
import numpy as np
import matplotlib.pyplot as plt
import h5py


class LabberData:
    def __init__(self, filePath=None, channelName=None, powerName: str = None, frequencyName=None, mode='resonator'):
        self._filePath = filePath

        self.h5data = h5py.File(self._filePath, 'r')

        self._channelName, self._channelMatrix, self._dataXMatrix, self._dataYMatrix = list(), None, None, None
        self._numData, self._channelNum = None, None

        # fit data is an NxM matrix (N is the number id)
        self._rawData, self._fitData, self._fitMask, self._fitID = None, None, None, None

        self._fitParamHistory, self._fitValueHistory = [], []
        self._fitErrorHistory = []
        self._configChosen = []
        self.mode, self.average = None, 0
        self._channelChosen = []

    def pull_data(self, mode='resonator', **kwargs):
        """
        Pull the data from labber file by certain frequency and power.
        :param mode: the pulling mode
        :param power:
        :param frequency:
        :return:
        """
        if mode == 'resonator':
            self.mode = mode
            # print(self.h5data['Data']['Data'][1,2,:])

            self._channelMatrix = self.h5data['Data']['Data'][:]
            self._dataXMatrix = self.h5data['Traces']['VNA - S21_t0dt'][:]
            self._dataYMatrix = self.h5data['Traces']['VNA - S21'][:]
            self._dataYMatrix = np.vectorize(complex)(self._dataYMatrix[:, 0, :], self._dataYMatrix[:, 1, :]).T
            # self._fitMask = np.ones((self._dataYMatrix.shape[1]), dtype=bool)

            if kwargs:
                t_x, t_y = self.mask_data(**kwargs)
                return t_x, t_y

        if mode == 'rabi':
            self.mode = mode
            self._dataXMatrix = self.h5data['Data']['Data'][:, 0, :].flatten()
            self._dataXMatrix *= 1e3
            self._dataYMatrix = self.h5data['Data']['Data'][:, 2, :].flatten()
            # self._dataYMatrix = self._dataYMatrix - self._dataYMatrix[-1]

            return self._dataXMatrix, self._dataYMatrix


    # Extract the needed data
    def mask_data(self, **kwargs):
        for i in range(len(self.h5data['Data']['Channel names'][:])):
            self._channelName.append(self.h5data['Data']['Channel names'][i][0].decode("utf-8"))
        self._numData = self.h5data['Traces']['VNA - S21_N'][0]

        self._channelNum = LabberData.get_channel_num(self._channelMatrix)
        fitIDlist = []

        # Check if specify repetition in kwargs, if not, will use average
        # Caution: Maybe we can use other words?
        if LabberData._find_str_args(self._channelName, 'rep') is not False:
            if 'repetition' not in kwargs:
                kwargs['repetition'] = 1
                self.average = 1

        for key in kwargs:
            current_channel_id = list()
            channelNo = LabberData._find_str_args(self._channelName, key)

            # Make the code work even while typing only one number, change into list
            if not hasattr(kwargs[key], '__iter__'):
                kwargs[key] = [kwargs[key]]

            for p in kwargs[key]:
                dataID = np.argwhere(self._channelMatrix == p)
                dataID = dataID[(dataID[:, 1] == channelNo)]
                dataID = (dataID[:, 2]) * self._channelNum[0] + dataID[:, 0]
                current_channel_id = current_channel_id + list(dataID)

            fitIDlist.append(current_channel_id)

        self._fitID = LabberData.find_common_list(fitIDlist)

        t_x = np.zeros((len(self._fitID), self._numData))
        t_y = np.zeros((len(self._fitID), self._numData)).astype('complex128')

        for i in range(len(self._fitID)):
            # Check if average?
            log_num = self._fitID[i]
            s21 = 0
            if self.average == 1:
                start = int(log_num // self._channelNum[0] * self._channelNum[0])
                s21 = np.mean(self._dataYMatrix[start:start + self._channelNum[0]], axis=0, dtype='complex128')

            else:
                s21 = self._dataYMatrix[log_num]

            t_x[i] = self.altspace(self._dataXMatrix[log_num], self._numData)
            t_y[i] = s21

        return t_x, t_y

    def fit_data(self, model='ResonatorModel', resonator_plot=False, method=None, verbose=None, **kwargs):
        if self.mode == 'resonator':
            self._dataXMatrix *= 1e-9  # Change to GHz
            self._channelChosen = self._get_channel_params_from_id(self._channelNum, self._fitID)
            # Check if average?
            for i in range(len(self._fitID)):
                log_num = self._fitID[i]
                s21 = 0
                if self.average == 1:
                    start = int(log_num // self._channelNum[0] * self._channelNum[0])
                    s21 = np.average(self._dataYMatrix[start:start+self._channelNum[0]], axis=0)

                else:
                    s21 = self._dataYMatrix[log_num]

                t_n = qf.QFit(self.altspace(self._dataXMatrix[log_num], self._numData), s21, model)
                t_n.guess()
                t_n.wash(method='fft')
                _sigma, _window = 0.05, 0.06
                if 'sigma' in kwargs:
                    _sigma = kwargs['sigma']
                if 'window' in kwargs:
                    _window = kwargs['window']

                if method is None:
                    pass
                elif method == 'lcaw':
                    t_n.wash('linecomp', window=_window)
                    t_n.add_weight(sigma=_sigma)
                elif method == 'lc':
                    t_n.wash('linecomp', window=_window)
                elif method == 'aw':
                    t_n.add_weight(sigma=_sigma)

                t_n.do_fit(verbose=verbose)
                self._fitParamHistory.append(t_n.fit_params())
                # self._fitValueHistory.append(t_n.fit_values())

                _freq = self._channelChosen[i][LabberData._find_str_args(self._channelName, 'freq')] * 1e-9
                _power = self._channelChosen[i][LabberData._find_str_args(self._channelName, 'power')]
                self._configChosen.append([_freq, _power])

                if resonator_plot:
                    t_n.polar_plot(id=log_num,
                                   f0=_freq,
                                   power=_power)

            if self.average == 1:
                print('Average is True')
            # self._fitValueHistory = np.array(self._fitValueHistory)

            return t_n

        if self.mode == 'rabi':
            t2 = qf.QFit(self._dataXMatrix, self._dataYMatrix, model='ExponentialModel')
            t2.add_models('LinearModel')
            t2.set_params('decay', 0.1)
            t2.do_fit(verbose=verbose)
            t2.pretty_print(plot_settings = {
                'x_label': 'Sequence duration (ms)',
                'y_label': 'Phase',
                'plot_title': 'Datasource',
                'fit_color': 'C4',
                'fig_size': (8, 6),
            })


    def push_data(self, filepath=None, filename=None, labber_path=None):
        if labber_path:
            sys.path.append(labber_path)
        try:
            import Labber
        except:
            print('No labber found here!')

        if self.mode == 'resonator':

            _LogFile = Labber.LogFile(self._filePath)

            if not filename:
                if not filepath:
                    filename = 'Fit_data_' + get_file_name_from_path(self._filePath)
                    filename = filename.replace('.hdf5', '')

                    filepath = 'D:/LabberDataTest'

            _Qi = np.repeat([x['Qi'] for x in self._fitParamHistory], 2) * 1e3
            # self._Qi = self._Qi.reshape((2 * len(self._fitEntry), -1))
            _Qe = np.repeat([x['Qe_mag'] for x in self._fitParamHistory], 2) * 1e3

            _FitLogFile = Labber.createLogFile_ForData(filename,
                                                       [dict(name='S21', unit='dB', vector=True, complex=True),
                                                        dict(name='Frequency', unit='GHz')],
                                                       [dict(name='Center Frequency', unit='GHz',
                                                             values=self._selectCenterFrequency),
                                                        dict(name='Output Power', unit='dB', values=self._selectPower),
                                                        dict(name='Qi', unit='', values=_Qi),
                                                        dict(name='Qe', unit='', values=_Qe),
                                                        ])

            for _entry in range(len(self._fitEntry)):
                _fitDictForAdd = _LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName]
                # _fitDictForAdd[self._channelName]['y'] = self._fitValueHistory[_entry]

                _fitDictForAdd = Labber.getTraceDict(self._fitValueHistory[_entry],
                                                     _LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName][
                                                         't0'],
                                                     _LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName][
                                                         't0'])

                # Add raw data
                _data = {'S21': self.S21[_entry],
                         'Frequency': self.frequency[_entry] * 1e-3,
                         }

                _FitLogFile.addEntry(_data)

                # Add fit data
                _data = {'S21': self._fitValueHistory[_entry],
                         'Frequency': self.frequency[_entry] * 1e-3,
                         }
                _FitLogFile.addEntry(_data)

            LabberData._data_structure_change(_FitLogFile.getFilePath(0), 3, self._Qi)
            LabberData._data_structure_change(_FitLogFile.getFilePath(0), 4, self._Qe)

    def _get_channel_params_from_id(self, num_list, id_list):
        self._channelChosen = self._channelMatrix[id_list % num_list[0], :, id_list // num_list[0]]
        print(self._channelChosen)
        return self._channelChosen

    def get_fit_params(self):
        return self._fitParamHistory

    def get_fit_config(self):
        return self._configChosen

    @staticmethod
    def _data_structure_change(path, row, data):
        """
        The aim here is to modify the h5 file to better display the fitting parameters.
        :param path: The h5 file path
        :param row: The column index in the ['Data']['Data']
        :return: None
        """
        h5 = h5py.File(path, 'r+')
        print(h5['Data']['Data'])
        h5['Data']['Data'][:, row, :] = data

    @staticmethod
    def altspace(t0dt, count, **kwargs):
        start = t0dt[0]
        step = t0dt[1]
        stop = start + (step * count)
        return np.linspace(start, stop, count, endpoint=False)

    @staticmethod
    def _find_str_args(choices: list, target, **kwargs):
        for i in range(len(choices)):
            if target in choices[i].lower():
                return i
        # print('Didn\'t find corresponding channel')
        return False

    @staticmethod
    def get_channel_num(channel_matrix):
        num_list = []
        for i in range(channel_matrix.shape[1]):
            num_list.append(len(np.unique(channel_matrix[:, i, :])))
        return num_list

    @staticmethod
    def find_common_list(id_list):
        common = id_list[0]
        if len(id_list) == 1:
            return common

        for i in id_list[1:]:
            common = np.intersect1d(common, i)
        return common


def get_file_name_from_path(path):
    import os
    head, tail = os.path.split(path)
    return tail



def fit_all_labber_resonator(file_loc, power=None, frequency=None, plot_all=False, attenuation=0):
    tn = qf.LabberData(file_loc)

    if not hasattr(power, '__iter__'):
        power = [power]
        # if power_limit:
        #     power = power[(power > power_limit[0]) & (power < power_limit[1])]
    power = np.sort(power)[::-1]  # Automatically sort in descend order

    freq_all, S21_all = tn.pull_data(power=power, frequency=frequency)
    # method with 'lc', 'aw', 'lcaw' or None

    Qi_list, Qi_err = [], []
    Qe_list, Qe_err = [], []
    for p in range(len(power)):
        print(power[p])
        _success = False
        for _win in range(0, 8, 2):
            freq, S21 = freq_all[p] * 1e-9, S21_all[p]
            t3 = qf.QFit(freq, S21, model='ResonatorModel')
            t3.guess()
            if _win != 0:
                t3.wash(method='complexcomp', window=_win * 1e-2)
            t3.wash(method='fft')
            t3.do_fit()
            qierr = t3.err_params('Qi')
            qeerr = t3.err_params('Qe_mag')

            # Check if fit fails?
            if t3.err_params('Qi') and t3.err_params('Qe_mag') is not None:
                if t3.fit_params('Qi') < 1e4:
                    if t3.err_params('Qi') < 0.5 * t3.fit_params('Qi'):
                        Qi_list.append(t3.fit_params('Qi') * 1e3)
                        Qi_err.append(qierr * 1e3)
                        Qe_list.append(t3.fit_params('Qe_mag') * 1e3)
                        Qe_err.append(qeerr * 1e3)
                        _success = True
                        if plot_all:
                            t3.polar_plot(power=power[p])
                        break

        if _success is False:
            freq, S21 = freq_all[p], S21_all[p]
            t3 = qf.QFit(freq, S21, model='ResonatorModel')
            t3.guess()
            t3.add_weight(sigma=0.15)
            t3.wash(method='fft')
            t3.do_fit()
            qierr = t3.err_params('Qi')
            qeerr = t3.err_params('Qe_mag')
            print(t3.fit_params('Qi'), qierr)
            if t3.err_params('Qi') and t3.err_params('Qe_mag') is not None:
                if t3.fit_params('Qi') < 1e4:
                    if t3.err_params('Qi') < 0.5 * t3.fit_params('Qi'):
                        Qi_list.append(t3.fit_params('Qi') * 1e3)
                        Qi_err.append(qierr * 1e3)
                        Qe_list.append(t3.fit_params('Qe_mag') * 1e3)
                        Qe_err.append(qeerr * 1e3)
                        if plot_all:
                            t3.polar_plot(power=power[p])
                            break

            Qi_list.append(0)
            Qi_err.append(0)
            Qe_list.append(0)
            Qe_err.append(0)
            print('Fails in fitting', power[p], 'dBm, assign 0,0 to Qi, Qe')

    fig, ax = plt.subplots()

    if attenuation:
        power -= attenuation

    ax.errorbar(power, Qi_list, yerr=Qi_err, fmt='o', c='r', label='Qi')
    ax.set_xlabel('Power(dBm)')
    ax.set_ylabel('$Q_{int}$', fontsize=14, c='r')
    ax2 = ax.twinx()
    ax2.errorbar(power, Qe_list, yerr=Qe_err, fmt='x', c='c', label='Qe')
    ax2.set_ylabel('$Q_{ext}$', fontsize=14, c='c')
    ax2.legend(loc="upper right")
    ax2.ticklabel_format(style='scientific', scilimits=(0, 3))
    ax.ticklabel_format(style='scientific',  scilimits=(0, 3))
    ax.legend(loc='upper left')
    ax.set_ylim(top=1.1 * max(Qi_list) if max(Qi_list) < 1e7 else 1e6, bottom=0.9 * min(Qi_list))
    ax2.set_ylim(top=1.1 * max(Qe_list) if max(Qe_list) < 1e7 else 1e6, bottom=0.9 * min(Qe_list))
    plt.title('Fit at ' + str(frequency * 1e-9)[:5] + ' GHz')
    plt.tight_layout()
    plt.show()


# Python program to illustrate reflection
def reverse(sequence):
    sequence_type = type(sequence)
    empty_sequence = sequence_type()

    if sequence == empty_sequence:
        return empty_sequence

    rest = reverse(sequence[1:])
    first_sequence = sequence[0:1]

    # Combine the result
    final_result = rest + first_sequence

    return final_result