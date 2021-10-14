import quantum_fitter as qf
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np

path = 'Z:/KianG/Resonator_Fitting_Test_Data/Al_on_sapphire_chip3/Power_sweeps_all_res_40dB_att_TWPA_on.hdf5'

qf.fit_all_labber_resonator(file_loc=path,
                            frequency=6.4972218e9,
                            power=list(range(-50, -40, 2)),  # Doesn't include 8
                            attenuation=40,
                            plot_all=False)
