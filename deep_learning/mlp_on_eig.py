import time

import h5py as h5

import climin.schedule

import climin.stops
import climin.initialize

from breze.learn.mlp import Mlp

f = h5.File('eigdata.hdf5', 'r')
X = f['matrices']
Z = f['eigvals']

VX = X
VZ = Z

max_passes = 100
batch_size = 2000
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size

stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)

optimizer = 'gd', {'steprate': 0.1}

m = Mlp(10000, [500], 1, hidden_transfers=['sigmoid'], out_transfer='identity', loss='squared',
        optimizer=optimizer, batch_size=batch_size)
climin.initialize.randomize_normal(m.parameters.data, 0, 1e-1)

losses = []
print 'max iter', max_iter

weight_decay = ((m.parameters.in_to_hidden**2).sum()
                + (m.parameters.hidden_to_out**2).sum())
weight_decay /= m.exprs['inpt'].shape[0]
m.exprs['true_loss'] = m.exprs['loss']
c_wd = 0.001
m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

start = time.time()
# Set up a nice printout.
keys = '#', 'seconds', 'loss', 'val loss'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)

for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
    if info['n_iter'] % n_report != 0:
        continue
    passed = time.time() - start
    losses.append((info['loss'], info['val_loss']))

    info.update({
        'time': passed})
    row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g\t%(train_emp)g\t%(val_emp)g' % info
    print row