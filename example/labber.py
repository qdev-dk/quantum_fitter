import Labber
import quantum_fitter as qf

t5 = qf.LabberData('../_resonator/VNA_all_res_v_pow_test.hdf5')
t5.pull_data(power=-30)
t5.fit_data()
t5.push_data()
