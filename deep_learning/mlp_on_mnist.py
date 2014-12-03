import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.schedule

import climin.stops
import climin.initialize

from breze.learn.mlp import Mlp
from breze.learn.data import one_hot

datafile = 'mnist.pkl.gz'
# Load data.

with gzip.open(datafile,'rb') as f:
    train_set, val_set, test_set = cPickle.load(f)

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

image_dims = 28, 28

max_passes = 150
batch_size = 250
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size

stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)

#optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.95, 'decay': 0.8}
optimizer = 'gd', {'steprate': 0.1}

m = Mlp(784, [800], 10, hidden_transfers=['sigmoid'], out_transfer='softmax', loss='cat_ce',
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

n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
f_n_wrong = m.function(['inpt', 'target'], n_wrong)

start = time.time()
# Set up a nice printout.
keys = '#', 'seconds', 'loss', 'val loss', 'train emp', 'val emp'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)

for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
    if info['n_iter'] % n_report != 0:
        continue
    passed = time.time() - start
    losses.append((info['loss'], info['val_loss']))

    #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
    #save_and_display(img, 'filters-%i.png' % i)
    info.update({
        'time': passed,
        'train_emp': f_n_wrong(X, Z),
        'val_emp': f_n_wrong(VX, VZ),
    })
    row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g\t%(train_emp)g\t%(val_emp)g' % info
    print row