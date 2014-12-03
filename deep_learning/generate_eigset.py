
import numpy as np
import qutip as qt
import h5py as h5

num_samples = 100000
dimensions = 100

f = h5.File('eigdata.hdf5','w')
matset = f.create_dataset("matrices", (num_samples, dimensions**2), dtype='complex64')
eigvalset = f.create_dataset("eigvals", (num_samples, 1), dtype='f')
eigvecset = f.create_dataset("eigvecs", (num_samples, dimensions), dtype='complex64')


for i in np.arange(num_samples):

    print 'generating samples %d' %(i+1)
    mat = qt.rand_herm(dimensions)
    eigval, eigvec = mat.groundstate()
    matset[i][...] = np.reshape(mat.full(),(dimensions**2,))
    eigvalset[i] = eigval
    eigvecset[i][...] = np.reshape(eigvec.full(),(dimensions,))

f.close()