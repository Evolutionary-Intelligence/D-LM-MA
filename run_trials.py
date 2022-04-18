import time
import os
import pickle
import argparse
import numpy as np

import pypoplib.continuous_functions as cf


class Experiment(object):
    def __init__(self, index, function, seed=None, distributed=None, island=None):
        self.index = index
        self.function = function
        self.seed = seed
        self.distributed = distributed
        self.island = island
        self.ndim_problem = 2000
        self._folder = 'pypop_benchmarks_lso'
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

    def run(self, optimizer):
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0 * np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0 * np.ones((self.ndim_problem,))}
        options = {'max_function_evaluations': np.Inf,
                   'max_runtime': 3600 * 2,  # seconds
                   'fitness_threshold': 1e-10,
                   'seed_rng': self.seed,
                   'record_fitness': True,
                   'record_fitness_frequency': 2000,
                   'verbose': False,
                   'sigma': 0.3} # for ES
        if self.distributed:
            options['n_islands'] = self.island
        solver = optimizer(problem, options)
        results = solver.optimize()
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function.__name__,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    def __init__(self, start, end, distributed, island):
        self.start = start
        self.end = end
        self.distributed = distributed
        self.island = island
        self.indices = range(self.start, self.end + 1)
        self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                          cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
        rng = np.random.default_rng(2021)
        self.seeds = rng.integers(np.iinfo(np.int64).max, size=(len(self.functions), 100))

    def run(self, optimizer):
        for index in self.indices:
            print('* experiment: {:d} ***:'.format(index))
            for d, f in enumerate(self.functions):
                start_time = time.time()
                print('  * function: {:s}:'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[d, index], self.distributed, self.island)
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


if __name__ == '__main__':
    start_run = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--optimizer', '-o', type=str)
    parser.add_argument('--distributed', '-d', type=bool, default=False)
    parser.add_argument('--island', '-i', type=int, default=100)
    args = parser.parse_args()
    params = vars(args)
    if params['optimizer'] == 'MAES':
        from maes import MAES as Optimizer
    elif params['optimizer'] == 'LMMAES':
        if params['distributed']:
            from distributed_lmmaes import DistributedLMMAES as Optimizer
        else:
            from lmmaes import LMMAES as Optimizer    
    experiments = Experiments(params['start'], params['end'], params['distributed'], params['island'])
    experiments.run(Optimizer)
    print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_run))
