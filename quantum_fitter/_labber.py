import sys
import quantum_fitter as qf
import numpy as np
import re
import os
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
        self.mode, self.average = None, 0
        self._channelChoosen = []

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
            print('what')
            if 'repetition' not in kwargs:
                kwargs['repetition'] = 1
                self.average = 1
                print('arrive here')

        for key in kwargs:
            channelNo = LabberData._find_str_args(self._channelName, key)

            # Make the code work even while typing only one number, change into list
            if not hasattr(kwargs[key], '__iter__'):
                kwargs[key] = [kwargs[key]]

            for p in kwargs[key]:
                dataID = np.argwhere(self._channelMatrix == p)
                dataID = dataID[(dataID[:, 1] == channelNo)]
                dataID = (dataID[:, 2]) * self._channelNum[0] + dataID[:, 0]
                fitIDlist.append(dataID)

        self._fitID = LabberData.find_common_list(fitIDlist)

        t_x = np.zeros((len(self._fitID), self._numData))
        t_y = np.zeros((len(self._fitID), self._numData))
        for id in self._fitID:
            t_x = self.altspace(self._dataXMatrix[id], self._numData)
            t_y = self._dataYMatrix[id]
        return t_x, t_y

    def fit_data(self, model='ResonatorModel', resonator_plot=False, method=None, **kwargs):
        self._dataXMatrix *= 1e-9  # Change to GHz
        from matplotlib import pyplot as plt
        self._channelChoosen = self._get_channel_params_from_id(self._channelNum, self._fitID)
        if model == 'ResonatorModel':

            # Check if average?
            for i in range(len(self._fitID)):
                id = self._fitID[i]
                s21 = 0
                if self.average == 1:
                    start = int(id // self._channelNum[0] * self._channelNum[0])
                    s21 = np.average(self._dataYMatrix[start:start+self._channelNum[0]], axis=0)

                else:
                    s21 = self._dataYMatrix[id]

                t_n = qf.QFit(self.altspace(self._dataXMatrix[id], self._numData), s21, model)
                t_n.guess()

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

                t_n.do_fit()
                self._fitParamHistory.append(t_n.fit_params())
                self._fitValueHistory.append(t_n.fit_values())
                if resonator_plot:
                    t_n.polar_plot(id=id,
                                   f0=self._channelChoosen[i][LabberData._find_str_args(self._channelName, 'freq')] * 1e-9,
                                   power=self._channelChoosen[i][LabberData._find_str_args(self._channelName, 'power')])
            self._fitValueHistory = np.array(self._fitValueHistory)
            return t_n

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
        _channelC = []
        for _ in id_list:
            _channelC.append(self._channelMatrix[id_list % num_list[0], :, id_list // num_list[0]].flatten())
        return _channelC

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
        print('Didn\'t find corresponding channel')
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
