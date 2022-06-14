import quantum_fitter.readout_tools as rdt
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, "example_data/ss_q1_rabi_v_ampl_5_fine.hdf5")


# Set the instence
rd = rdt.Readout(file_path, size=200, verbose=0)

# Set the forloop
set_list = list(range(50, 101, 10)) + [150, 200, 400, 600]
for i in tqdm(set_list):
    rd.set_dataset_size(size=i)
    rd.set_plot_dir(i, param_name=" Dataset size", score_name=None)

# Plots figuer
rd.plot_param_effect()

