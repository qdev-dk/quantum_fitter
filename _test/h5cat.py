from h5glance import H5Glance
import h5py

with h5py.File('BFC3-16-Qu3_T1_quick.hdf5', 'r+') as f:
    H5Glance(f)
    print(H5Glance(f))
