import sys
import Labber
from Labber import ScriptTools
import quantum_fitter as qf
import numpy as np
import re
import os
import h5py


class LabberData:
    def __init__(self, file_path=None, channel=None, power: str = None, frequency=None):
        self.file_path = file_path
        self._LogFile = Labber.LogFile(file_path)
        self.h5data = h5py.File(file_path, 'r')
        print(list(self.h5data['Data']['Data'].shape))
        self.h5data.close()
        self._EntryName = list(self._LogFile.getEntry().keys())

        self._channelName = channel if channel else self._LogFile.getLogChannels()[0].get('name')
        self._powerName = list(power) if power else \
            list(filter(lambda v: re.match('.*[Pp]ower.*$', v), self._LogFile.getEntry().keys()))
        self._frequencyName = list(frequency) if power else \
            list(filter(lambda v: re.match('.*[Ff]req.*$', v), self._LogFile.getEntry().keys()))

        self._powerValueList = self._LogFile.getData(self._powerName[0]).flatten()
        print(self._LogFile.getData(self._powerName[0]).shape)
        self._freqValueList = self._LogFile.getData(self._frequencyName[0]).flatten() * 1e-6

        self._runPath = os.path.dirname(os.path.abspath(__file__))
        self._fitEntry = None
        self._fitParamHistory = []
        self._fitValueHistory = []

        self.frequency, self.S21 = None, None

    def pull_data(self, power=None, frequency=None):
        if power:
            self._fitEntry = np.argwhere(self._powerValueList == power).flatten()
        elif frequency:
            self._fitEntry = np.argwhere(self._freqValueList == frequency).flatten()

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
            t_n.filter_guess(level=11)
            t_n.do_fit()
            self._fitParamHistory.append(t_n.fit_params())
            self._fitValueHistory.append(t_n.fit_values())
        self._fitValueHistory = np.array(self._fitValueHistory)

    def push_data(self):
        for _entry in range(len(self._fitEntry)):
            print(_entry)
            _fitDictForAdd = self._LogFile.getEntry(entry=self._fitEntry[_entry])
            print(self._LogFile.getEntry(entry=self._fitEntry[_entry]))
            _fitDictForAdd[self._channelName]['y'] = self._fitValueHistory[_entry]
            print(_fitDictForAdd.keys())
            self._LogFile.addEntry(_fitDictForAdd)
            print(self._LogFile.getEntry(entry=self._fitEntry[_entry]))
            print(self._LogFile.getData(self._powerName[0]).shape)
            print(self._LogFile.getData(self._powerName[0]))
            # self.h5data = h5py.File(self.file_path, 'r')
            # print(list(self.h5data['Traces']['VNA - S21_N']))
            # self.h5data.close()
            exit()
        pass



