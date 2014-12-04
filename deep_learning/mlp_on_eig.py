import time

import h5py as h5
import cPickle as cp

import climin.schedule

import climin.stops
import climin.initialize

from breze.learn.mlp import Mlp
from sklearn.grid_search import ParameterSampler


def draw_pars(n=1):
    class OptimizerDistribution(object):
        def rvs(self):
            grid = {
            'step_rate': [0.0001, 0.0005, 0.005],
            'momentum': [0.99, 0.995],
            'decay': [0.9, 0.95],
            }
            sample = list(ParameterSampler(grid, n_iter=1))[0]
            sample.update({'step_rate_max': 0.05, 'step_rate_min': 1e-5})
            return 'rmsprop', sample
    grid = {
    'n_hidden': [500, 500],
    'hidden_transfer': ['sigmoid', 'tanh', 'rectifier'],
    'par_std': [1.5, 1, 1e-1, 1e-2],
    'optimizer': OptimizerDistribution(),
    }

    sampler = ParameterSampler(grid, n)
    return sampler

def run_mlp(n_job, pars):

    f = h5.File('eigdata.hdf5', 'r')
    X = f['matrices'][...]
    Z = f['eigvals'][...]

    f = open('mlp_training_%d' %n_job, 'w')

    max_passes = 100
    batch_size = 2000
    max_iter = max_passes * X.shape[0] / batch_size
    n_report = X.shape[0] / batch_size

    stop = climin.stops.AfterNIterations(max_iter)
    pause = climin.stops.ModuloNIterations(n_report)

    m = Mlp(20000, pars['n_hidden'], 1, hidden_transfers=[pars['hidden_transfer']]*len(pars['n_hidden']), out_transfer='identity', loss='squared',
            optimizer=pars['optimizer'], batch_size=batch_size)
    climin.initialize.randomize_normal(m.parameters.data, 0, pars['par_std'])

    losses = []
    f.write('max iter: %d \n' %max_iter)

    weight_decay = ((m.parameters.in_to_hidden**2).sum()
                    + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = 0.001
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    start = time.time()
    # Set up a nice printout.
    keys = '#', 'seconds', 'loss', 'val_loss'
    max_len = max(len(i) for i in keys)
    header = '\t'.join(i for i in keys)
    f.write(header + '\n')
    f.write(('-' * len(header)) + '\n')

    for i, info in enumerate(m.powerfit((X, Z), (X, Z), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        passed = time.time() - start
        losses.append((info['loss'], info['val_loss']))

        info.update({
            'time': passed})
        row = '%(n_iter)i\t%(time)g\t%(loss)g\t%(val_loss)g' % info
        f.write(row)

    f.write('best val_loss: %f \n' %info['best_loss'])
    f.close()

    cp.dump(info['best_pars'], open('best_pars_%d.pkl' %n_job, 'w'))