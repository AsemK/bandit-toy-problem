import pickle
import time
from bayes_opt import BayesianOptimization
from RandomSurface import RandomSurface
import numpy as np

variables = [('a', (0, 1)), ('b', (0, 1)), ('c', (0, 1)), ('d', (0, 1)), ('e', (0, 1)), ('f', (0, 1)), ('g', (0, 1)), ('h', (0, 1))]


def target(surface, **params):
    keys = list(params.keys())
    keys.sort()
    values = []
    for key in keys:
        values.append(params[key])
    return surface.get_surface_at(values)


dims = [2, 3, 4, 5, 6]
init = 2
for dim in dims:
    with open('surfaces' + str(dim) + '.pkl', 'rb') as f:
        surfaces = pickle.load(f)
        f.close()
    with open('initpts' + str(dim)*init + '.pkl', 'rb') as f:
        initial_points = pickle.load(f)
        f.close()
    init_n = initial_points[0].shape[0]
    iter_n = 40
    surf_n = len(surfaces)
    acq = 'ucb'
    consts = [1]
    for const in consts:
        print('-------------setting' + str(const))
        start = time.time()
        results = np.empty((surf_n, init_n + iter_n))
        points = []
        for i in range(surf_n):
            print('-------------surface' + str(i))
            bo = BayesianOptimization(lambda **params: target(surfaces[i], **params), dict(variables[0:dim]))
            for j in range(init_n):
                bo._observe_point(initial_points[i][j])
            bo.maximize(init_points=0, n_iter=iter_n, acq=acq, kappa=const, xi=const, alpha=1e-5)
            results[i] = bo.space.Y
            points.append(bo.space.X)
        end = time.time()
        print('time (s): ' + str(end-start))
        with open('res-' + str(dim)*init + '-' + acq + '-' + str(const) + '.pkl', 'wb') as f:
            pickle.dump((results, points, init_n, iter_n, acq, const), f)