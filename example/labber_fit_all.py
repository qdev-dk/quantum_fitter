import quantum_fitter as qf
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

path = 'Z:/KianG/Resonator_Fitting_Test_Data/Al_on_sapphire_chip3/Power_sweeps_all_res_40dB_att_TWPA_on.hdf5'

qf.fit_all_labber_resonator(file_loc=path,
                            frequency=5.712953e9,
                            power=list(range(6, 8, 2)),  # Doesn't include 8
                            plot_all=False)
