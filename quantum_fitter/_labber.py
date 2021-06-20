import sys
import Labber
from Labber import ScriptTools
import quantum_fitter as qf
import numpy as np
import re
import os
import h5py


class LabberData:
    def __init__(self, filePath=None, channelName=None, powerName:str=None, frequencyName=None, mode='VNA'):
        self.file_path = filePath
        self._LogFile = Labber.LogFile(filePath)
        self.h5data = h5py.File(filePath, 'r')
        self.h5data.close()
        self._EntryName = list(self._LogFile.getEntry().keys())

        self._channelName = channelName if channelName else self._LogFile.getLogChannels()[0].get('name')
        self._powerName = list(powerName) if powerName else \
            list(filter(lambda v: re.match('.*[Pp]ower.*$', v), self._LogFile.getEntry().keys()))
        self._frequencyName = list(frequencyName) if powerName else \
            list(filter(lambda v: re.match('.*[Ff]req.*$', v), self._LogFile.getEntry().keys()))

        self._powerValueList = self._LogFile.getData(self._powerName[0]).flatten()
        print(self._LogFile.getData(self._powerName[0]).shape)
        self._freqValueList = self._LogFile.getData(self._frequencyName[0]).flatten() * 1e-6
        self._centerFrequencyList = self._LogFile.getData('VNA - Center frequency') * 1e-9

        self._runPath = os.path.dirname(os.path.abspath(__file__))
        self._fitEntry = None
        self._fitParamHistory = []
        self._fitValueHistory = []

        self.frequency, self.S21 = None, None
        self.Qi, self.Qe = None, None

        self._selectPower = None
        self._selectCenterFrequency = None

    def pull_data(self, power=None, frequency=None):
        if power:
            self._fitEntry = np.argwhere(self._powerValueList == power).flatten()
            # leave an option for more selections in power (not yet complete)
            if self._selectPower is None:
                self._selectPower = np.repeat(power, 2 * len(self._fitEntry))
                # self._selectPower = np.array(power)
                self._selectCenterFrequency = self._centerFrequencyList[:, 0:2].flatten()
            else:
                self._selectPower = np.vstack(self._selectPower, np.repeat(power, len(self._fitEntry)))

        elif frequency:
            self._fitEntry = np.argwhere(self._freqValueList == frequency).flatten()

        else:
            print('You need to specify either power or frequency')
            return

        freq = []
        S21 = []

        for e in self._fitEntry:
            [_xData, _yData] = self._LogFile.getTraceXY(y_channel=self._channelName, entry=e)
            freq.append(_xData)
            S21.append(_yData)

        self.frequency = np.array(freq) * 1E-6
        self.S21 = np.array(S21)
        return self.frequency, self.S21

    def fit_data(self, model='ResonatorModel'):
        for thing in range(len(self.frequency)):
            t_n = qf.QFit(self.frequency[thing], self.S21[thing], model)
            t_n.filter_guess(level=5)
            t_n.do_fit()
            self._fitParamHistory.append(t_n.fit_params())
            self._fitValueHistory.append(t_n.fit_values())
        self._fitValueHistory = np.array(self._fitValueHistory)
        return t_n

    def polar_plot(self):
        _tn = self.fit_data()
        _tn.polar_plot(power=self._selectPower[0], suptitle=get_file_name_from_path(self.file_path))

    def push_data(self, filepath=None, filename=None):
        if not filename:
            if not filepath:
                filename = 'Fit_data_'+get_file_name_from_path(self.file_path)
                filepath = 'D:/LabberDataTest'

        self._Qi = np.repeat([x['Qi'] for x in self._fitParamHistory], 2)
        self._Qe = np.repeat([x['Qe_mag'] for x in self._fitParamHistory], 2)

        _FitLogFile = Labber.createLogFile_ForData(filename,
        [dict(name='S21', unit='dB', vector=True, complex=True),
        dict(name='Frequency', unit='GHz')],
        [dict(name='Center Frequency', unit='GHz', values=self._selectCenterFrequency, vector=True),
         dict(name='Output Power', unit='dB', values=self._selectPower, vector=True),
        ])

        for _entry in range(len(self._fitEntry)):
            _fitDictForAdd = self._LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName]
            # _fitDictForAdd[self._channelName]['y'] = self._fitValueHistory[_entry]

            _fitDictForAdd = Labber.getTraceDict(self._fitValueHistory[_entry],
                            self._LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName]['t0'],
                         self._LogFile.getEntry(entry=self._fitEntry[_entry])[self._channelName]['t0'])

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


def get_file_name_from_path(path):
    import os
    head, tail = os.path.split(path)
    return tail