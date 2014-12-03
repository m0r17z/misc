
import numpy as np
import qutip as qt
import h5py as h5

f = h5.File('eigdata.hdf5','w')
matset = f.create_dataset("matrices", (1000, 10000), dtype='c')
eigvalset = f.create_dataset("eigvals", (1000, 1), dtype='f')
eigvecset = f.create_dataset("eigvecs", (1000, 100), dtype='c')


for i in np.arange(1000):

    mat = qt.rand_herm(100)
    eigval, eigvec = mat.groundstate()
    matset[i][...] = np.reshape(mat.full(),(10000,))
    eigvalset[i] = eigval
    eigvecset[i][...] = np.reshape(eigvec.full(),(100,))

f.close()