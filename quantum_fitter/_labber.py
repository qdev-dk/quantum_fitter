import sys
import Labber
from Labber import ScriptTools
import quantum_fitter as qf
import numpy as np
import re
import os

class LabberData:
    def __init__(self, file_path=None, channel=None, power: str = None, frequency=None):
        self._LogFile = Labber.LogFile(file_path)
        self._EntryName = list(self._LogFile.getEntry().keys())

        self._channelName = channel if channel else self._LogFile.getLogChannels()[0].get('name')
        self._powerName = list(power) if power else \
            list(filter(lambda v: re.match('.*[Pp]ower.*$', v), self._LogFile.getEntry().keys()))
        self._frequencyName = list(frequency) if power else \
            list(filter(lambda v: re.match('.*[Ff]req.*$', v), self._LogFile.getEntry().keys()))

        self._powerValueList = self._LogFile.getData(self._powerName[0]).flatten()
        self._freqValueList = self._LogFile.getData(self._frequencyName[0]).flatten() * 1e-6

        self._runPath = os.path.dirname(os.path.abspath(__file__))
        self._fitParamHistory = []
        self._fitValueHistory = []

        self.frequency, self.S21 = None, None

    def pull_data(self, power=None, frequency=None):
        if power:
            _entry = np.argwhere(self._powerValueList == power).flatten()
        elif frequency:
            _entry = np.argwhere(self._freqValueList == frequency).flatten()

        freq = []
        S21 = []

        for e in _entry:
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

    def push_data(self):
        self._LogFile.addEntry(self._fitValueHistory[0])
        pass


