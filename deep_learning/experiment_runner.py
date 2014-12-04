__author__ = 'august'

import mlp_on_eig as mlp

from sklearn.grid_search import ParameterSampler

nr_jobs = 0

while nr_jobs < 16:
    pars = mlp.draw_pars()
    mlp.run_mlp(nr_jobs, pars)
    nr_jobs += 1
