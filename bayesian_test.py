from RandomSurface import RandomSurface
import pickle
import numpy as np
import pyDOE
import util

surf_n = 10
init_n = 10
dim = 6

surfaces = []
for i in range(surf_n):
    problem = RandomSurface([(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], cov_diag_range=(0.005, 0.01), cov_off_diag_range=(-0.005, 0.005))
    problem.create_random_surface(30)
    surfaces.append(problem)

with open('surfaces' + str(dim) + '.pkl', 'wb') as f:
    pickle.dump(surfaces, f)

initial_points = []
for surface in surfaces:
    init_pts = np.empty((init_n, surface.get_dim()))
    for i in range(init_n):
        init_pt = np.random.random(surface.get_dim())
        init_pts[i] = util.scale_vec(init_pt, surface.domains)
    initial_points.append(init_pts)

with open('initpts' + str(dim) + '.pkl', 'wb') as f:
    pickle.dump(initial_points, f)

initial_points = []
for i in range(surf_n):
    init_pts = pyDOE.lhs(dim, init_n, 'center')
    initial_points.append(init_pts)

with open('initpts' + str(dim) + str(dim) + '.pkl', 'wb') as f:
    pickle.dump(initial_points, f)

