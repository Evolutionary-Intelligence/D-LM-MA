import numpy as np
import ray

from pypoplib.continuous_functions import load_shift_and_rotation as load_sr
from pypoplib.es import ES


class DistributedES(ES):
    """Distributed (Meta-) Evolution Strategies (DES).
    """
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self._customized_class = None  # set by the wrapper
        self.n_islands = options.get('n_islands')  # number of inner ESs
        self.island_max_runtime = options.get('island_max_runtime', 3 * 60)  # for inner ESs
        self.n_better_islands = int(np.maximum(20, self.n_islands / 10))  # for outer ES
        w_base, w = np.log((self.n_better_islands * 2 + 1) / 2), np.log(np.arange(self.n_better_islands) + 1)
        self._dw = (w_base - w) / (self.n_better_islands * w_base - np.sum(w))
        self.max_runtime = options.get('max_runtime', 3600 * 2)
        # to ensure that actual runtime does not exceed 'max_runtime' as much as possible
        self.max_runtime -= (self.island_max_runtime + 60)
        # for outer ES (or for online hyper-parameter optimization)
        self.sigma_scale = [0.5, 1, 1.5, 2, 5, 10, 20]
        self.learning_ratios = np.linspace(0, 1, 11)
        self.sl = []
        for i in self.sigma_scale:
            for j in self.learning_ratios:
                self.sl.append([i, j])

    def optimize(self, fitness_function=None, args=None):  # for all iterations (generations)
        super(ES, self).optimize(fitness_function)
        ray.init(address='auto', runtime_env={'py_modules': ['./pypoplib']})
        ray_problem = ray.put(self.problem)
        shift_vector, rotation_matrix = load_sr(self.fitness_function, np.empty((self.ndim_problem,)))
        ray_args = ray.put({'shift_vector': shift_vector, 'rotation_matrix': rotation_matrix})
        ray_island = ray.remote(self._customized_class)
        fitness = []  # store all fitness generated during search
        is_first_generation = True
        x = self.rng_initialization.uniform(self.initial_lower_boundary,
                                            self.initial_upper_boundary,
                                            size=(self.n_islands, self.ndim_problem))
        best_x, best_y = np.zeros((self.n_islands, self.ndim_problem)), np.zeros((self.n_islands,))
        n_evolution_paths = 4 + int(3 * np.log(self.ndim_problem))
        s = np.zeros((self.n_islands, self.ndim_problem))  # for mutation strengths of all inner ESs
        tm = np.zeros((self.n_islands, n_evolution_paths, self.ndim_problem))  # transform matrices of all inner ESs
        c_s = (2 * self.n_individuals / self.ndim_problem) * np.ones((self.n_islands,))
        sigmas = 0.3 * np.ones((self.n_islands,))  # mutation strengths of all inner ESs
        n_low_dim = 4 * len(self.sigma_scale) + 2 * len(self.learning_ratios)
        while not self._check_terminations():
            order = np.argsort(best_y)
            index_1, index_2, index_3, index_4, index_5, index_6 = 0, 0, 0, 0, 0, 0
            ray_options, ray_islands, ray_results = [], [], []
            w_x = np.zeros((self.ndim_problem,))
            w_s = np.zeros((self.ndim_problem,))
            w_tm = np.zeros((n_evolution_paths, self.ndim_problem))
            w_c_s = 0
            w_sigma = 0
            for k in range(self.n_better_islands):
                w_x += self._dw[k] * best_x[order[k]]
                w_s += self._dw[k] * s[order[k]]
                w_tm += self._dw[k] * tm[order[k]]
                w_c_s += self._dw[k] * c_s[order[k]]
                w_sigma += self._dw[k] * sigmas[order[k]]
            for p in range(self.n_islands):
                index = order[-(p + 1)]
                best_x[index], s[index], tm[index], c_s[index], sigmas[index] = w_x, w_s, w_tm, w_c_s, w_sigma
                if p < len(self.sigma_scale):
                    index_1 += 1
                    sigmas[index] = w_sigma * self.sigma_scale[index_1 - 1]
                elif p < 2 * len(self.sigma_scale):
                    index_2 += 1
                    sigmas[index] = w_sigma * self.sigma_scale[index_2 - 1]
                elif p < 3 * len(self.sigma_scale):
                    index_3 += 1
                    tm[index] = np.zeros((n_evolution_paths, self.ndim_problem))
                    sigmas[index] = w_sigma * self.sigma_scale[index_3 - 1]
                elif p < 4 * len(self.sigma_scale):
                    index_4 += 1
                    tm[index] = np.zeros((n_evolution_paths, self.ndim_problem))
                    sigmas[index] = w_sigma * self.sigma_scale[index_4 - 1]
                elif p < 4 * len(self.sigma_scale) + len(self.learning_ratios):
                    index_5 += 1
                    c_s[index] = self.learning_ratios[index_5 - 1]
                elif p < 4 * len(self.sigma_scale) + 2 * len(self.learning_ratios):
                    index_6 += 1
                    c_s[index] = self.learning_ratios[index_6 - 1]
                elif p < n_low_dim + len(self.sl):
                    pp = p - n_low_dim
                    sigmas[index] = w_sigma * self.sl[pp][0]
                    c_s[index] = self.sl[pp][1]
                if is_first_generation:  # only for the first generation
                    best_x[p] = x[p]  # each island is initialized randomly
            for p in range(self.n_islands):  # in parallel
                options = {'x': best_x[p],
                           'max_runtime': self.island_max_runtime,
                           'fitness_threshold': self.fitness_threshold,
                           'seed_rng': self.rng_optimization.integers(0, np.iinfo(np.int64).max),
                           'verbose': False,
                           'record_fitness': True,
                           'record_fitness_frequency': 1,
                           's': s[p],
                           'tm': tm[p],
                           'c_s': c_s[p],
                           'sigma': sigmas[p],
                           }
                ray_options.append(options)
                ray_islands.append(ray_island.remote(ray_problem, ray_options[p]))
                ray_results.append(ray_islands[p].optimize.remote(self.fitness_function, ray_args))
            results = ray.get(ray_results)
            is_first_generation = False
            for p, r in enumerate(results):
                if self.best_so_far_y > r['best_so_far_y']:
                    self.best_so_far_y = np.copy(r['best_so_far_y'])
                    self.best_so_far_x = np.copy(r['best_so_far_x'])
                best_y[p] = r['best_so_far_y']
                best_x[p] = r['mean']
                s[p] = r['s']
                tm[p] = r['tm']
                c_s[p] = r['c_s']
                sigmas[p] = r['sigma']
                self.n_function_evaluations += r['n_function_evaluations']
                self.time_function_evaluations += r['time_function_evaluations']
                if self.record_fitness:
                    fitness.extend(r['fitness'][:, 1])
        ray.shutdown()
        return self._collect_results(fitness)
