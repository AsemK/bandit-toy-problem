import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
from ContinuumBanditProblem import ContinuumBanditProblem
from RandomSurface import RandomSurface
import util


class ToyContinuumBandit(ContinuumBanditProblem):
    """
    A simple wrapper to use RandomSurface ad a ContinuumBanditProblem
    """
    def __init__(self, domains, values_range=(0.0, 1.0), cov_diag_range=(0.01, 0.02),
                 cov_off_diag_range=(-0.01, 0.01), smooth=True, rocks=30):
        self.random_surface = RandomSurface(domains, values_range, cov_diag_range,
                                            cov_off_diag_range, smooth)
        self.random_surface.create_random_surface(rocks)

    def get_domains(self):
        return self.random_surface.get_domains()

    def reward(self, params):
        return self.random_surface.get_surface_at(params)


class GPUCB():
    def __init__(self, problem):
        self.problem = problem
        self.N = 10
        self.sigma_s2 = 0.000005
        self.sigma_a2 = 100
        self.dim = len(self.problem.get_domains())
        self.W = np.ones(self.dim)
        self.m = 0.3

    def initialize(self):
        self.Y = np.random.random((self.N, self.dim))
        self.t = self.problem.reward(util.scale_vec(self.Y, self.problem.get_domains()))

    def mean(self, x, alpha):
        return ((self.sigma_a2*util.gaussian_kernel(x, self.Y, 0.5, self.W)).reshape(1, -1).dot(alpha)).reshape(1,)

    def std(self, x, gamma):
        var = self.sigma_a2*util.gaussian_kernel(x, x, 0.5, self.W)
        for i in range(self.N):
            for j in range(self.N):
                var -= self.sigma_a2*util.gaussian_kernel(x, 0.5*(self.Y[i]+self.Y[j]),
                                                          1, self.W)*gamma[i, j]
        return np.sqrt(abs(var))

    def loop(self):
        M_best = 0
        x_best = None

        K2 = np.empty((self.N, self.N))
        for i in range(self.N):
            K2[i, :] = util.gaussian_kernel(self.Y[i], self.Y, 0.25, self.W)
        K = self.sigma_a2*K2*K2

        alpha = np.linalg.inv(K + self.sigma_s2*np.identity(self.N))
        gamma = alpha*K2
        alpha = alpha.dot(self.t.reshape((-1,1)))

        for k in range(self.N):
            x_old = 999999*np.ones(self.dim)
            x = self.Y[k]
            while (x-x_old).dot(x-x_old) > 0.0001 and np.sqrt((x-self.Y[k]).dot(x-self.Y[k])) <= self.m:
                if self.problem.is_valid_params(x):
                    x_old = x
                else:
                    x = x_old
                    break
                mean = self.mean(x, alpha)
                std = self.std(x, gamma)
                d_mean = sum(np.diag(self.W).dot((self.Y[i] - x_old)) * self.sigma_a2
                             * util.gaussian_kernel(x_old, self.Y[i], 0.5, self.W)
                             * alpha[i] for i in range(self.N))
                d_std = sum(2.0 * gamma[i, j] / std * np.diag(self.W).dot(0.5*(self.Y[i]+self.Y[j])-x_old)
                            * self.sigma_a2 * util.gaussian_kernel(x_old,0.5*(self.Y[i]+self.Y[j]), 1, self.W)
                            for i in range(self.N) for j in range(self.N))
                x = x_old + 0.005*(d_mean + d_std)

            M = self.mean(x, alpha) + self.std(x, gamma)
            # print(x)
            # print(M)
            if M > M_best:
                M_best = M
                x_best = x

        self.Y = np.concatenate((self.Y, x_best[np.newaxis, :]))
        self.t = np.concatenate((self.t, self.problem.reward(x_best).reshape((1,))))
        self.N += 1
        print('--------iteration end----------------')
        print(x_best)
        print(M_best)
        print(self.t[-1])
        print(max(self.t))


if __name__ == '__main__':
    problem = ToyContinuumBandit([(0, 1), (0, 1)], cov_diag_range=(0.01, 0.02), cov_off_diag_range=(-0.01, 0.01))
    solver = GPUCB(problem)
    solver.initialize()
    print(solver.Y)
    print(solver.t)
    rew = [max(solver.t)]
    for i in range(25):
        print('--------------------------------')
        solver.loop()
        rew += [max(solver.t)]
    problem.random_surface.contour()
    plt.figure()
    plt.plot(rew,'.r')
    plt.show()

