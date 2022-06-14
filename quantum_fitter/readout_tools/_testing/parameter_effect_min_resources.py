import quantum_fitter.readout_tools as rdt
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

# Set up path
dir = os.path.dirname(__file__)
file_path = os.path.join(dir, "example_data/ss_q1_rabi_v_ampl_5_fine.hdf5")


# Set the instence
rd = rdt.Readout(file_path, size=2000, verbose=1)

# Set the forloop
set_list = np.logspace(3, 8, num=11, base=2.5).astype(int)

for i in tqdm(set_list):
    rd._min_resources = i
    rd.do_fit()
    rd.set_plot_dir(
        i, param_name="Min_resources (number of single shot)", score_name=None
    )

    rd.plot_cv_iterations(
        title=f"{rd._get_file_name_from_path(rd._filePath)}, Min_resources: {i}"
    )
    plt.show()

# Plots figuer
rd.plot_param_effect()
