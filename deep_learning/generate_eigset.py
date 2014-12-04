
import numpy as np
import qutip as qt
import h5py as h5

num_samples = 100000
dimensions = 100

f = h5.File('eigdata.hdf5','w')
matset = f.create_dataset("matrices", (num_samples, 2*(dimensions**2)), dtype='f')
eigvalset = f.create_dataset("eigvals", (num_samples, 1), dtype='f')
eigvecset = f.create_dataset("eigvecs", (num_samples, 2*dimensions), dtype='f')


for i in np.arange(num_samples):

    print 'generating samples %d' %(i+1)
    mat = qt.rand_herm(dimensions)
    eigval, eigvec = mat.groundstate()
    mat = np.reshape(mat.full(), (dimensions**2,))
    real_mat = np.real(mat)
    imag_mat = np.imag(mat)
    eigvec = np.reshape(eigvec.full(), (dimensions,))
    real_eigvec = np.real(eigvec)
    imag_eigvec = np.imag(eigvec)
    matset[i][...] = np.append(real_mat, imag_mat)
    eigvalset[i] = eigval
    eigvecset[i][...] = np.append(real_eigvec, imag_eigvec)

f.close()