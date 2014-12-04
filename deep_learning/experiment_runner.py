__author__ = 'august'

import mlp_on_eig as mlp

nr_jobs = 0

while nr_jobs < 16:
    mlp.run_mlp()
    nr_jobs += 1
