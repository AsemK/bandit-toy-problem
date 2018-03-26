import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

class RandomSurface:
    def __init__(self, pos_ranges, values_range=(0.0, 1.0), cov_diag_range=(0, 0.1), cov_off_diag_range=(-1, 1)):
        self.pos_ranges = pos_ranges
        self.values_range = values_range
        self.cov_diag_range = cov_diag_range
        self.cov_off_diag_range = cov_off_diag_range

        self.rocks = []
        self.get_surface_at = self.get_smooth_surface_at

    def get_dim(self):
        return len(self.pos_ranges)

    def get_rocks_num(self):
        return len(self.rocks)

    def get_values_range(self):
        return self.values_range

    @staticmethod
    def random_positive_definite_mat(dim, diag_range, off_diag_range):
        Q = np.zeros([dim, dim])
        while np.linalg.matrix_rank(Q) < dim:
            Q = np.random.random([dim, dim])*(off_diag_range[1]-off_diag_range[0])+off_diag_range[0]
        M = np.zeros([dim, dim])
        while not np.all(np.linalg.eigvals(M) > 0):
            M = (Q.transpose().dot(np.diag(np.random.random(dim)*(diag_range[1]-diag_range[0])+diag_range[0]))).dot(Q)
        return M

    def random_gaussian_surface(self):
        mean_pos = [np.random.random()*(self.pos_ranges[i][1]-self.pos_ranges[i][0])+self.pos_ranges[i][0]
                    for i in range(self.get_dim())]
        cov = self.random_positive_definite_mat(self.get_dim(), self.cov_diag_range, self.cov_off_diag_range)
        rv = multivariate_normal(mean_pos, cov)
        return rv, mean_pos, cov

    def create_random_surface(self, rocks):
        heights = np.random.random(rocks) * (self.values_range[1] - self.values_range[0]) + self.values_range[0]
        heights *= self.values_range[1]/max(heights)
        self.rocks = []
        for i in range(rocks):
            self.rocks.append((*self.random_gaussian_surface(), heights[i]))

    def get_rough_surface_at(self, pos):
        result = 0
        for rock in self.rocks:
            result = np.maximum(result, rock[0].pdf(pos)/rock[0].pdf(rock[1])*rock[-1])
        return result

    def get_smooth_surface_at(self, pos):
        result = 0
        for rock in self.rocks:
            result += rock[0].pdf(pos)/rock[0].pdf(rock[1])*rock[-1]
        return result

    def graph(self, axis, other_axes_values, precision=0.01):
        # should rise an error
        if self.get_dim() < 1 or self.get_dim() < axis:
            return

        x = np.arange(*self.pos_ranges[axis], precision)
        pos = np.zeros((len(x), self.get_dim()))
        pos[:, axis] = x
        c = 0
        for i in range(self.get_dim()):
            if i == axis:
                pos[:, axis] = x
            else:
                pos[:, i] = other_axes_values[c]*np.ones(x.shape)
                c += 1

        plt.figure()
        plt.plot(x, self.get_surface_at(pos))
        plt.show()

    def contour(self, axis1, axis2, other_axes_values, precision=0.01):
        # should rise an error
        if self.get_dim() < 2 or self.get_dim() < axis1 or self.get_dim() < axis2:
            return

        x, y = np.meshgrid(np.arange(*self.pos_ranges[axis1], precision), np.arange(*self.pos_ranges[axis2], precision))
        pos = np.empty(x.shape + (self.get_dim(),))
        c = 0
        for i in range(self.get_dim()):
            if i == axis1:
                pos[:, :, i] = x
            elif i == axis2:
                pos[:, :, i] = y
            else:
                pos[:, :, i] = other_axes_values[c] * np.ones(x.shape)
                c += 1

        plt.figure()
        plt.contourf(x, y, self.get_surface_at(pos), levels=np.arange(*self.get_values_range(), 0.1))
        plt.axis('square')
        plt.axis([*self.pos_ranges[axis1], *self.pos_ranges[axis2]])
        plt.show()


if __name__ == '__main__':
    # random_surface = RandomSurface([(0.5, 1.5), (0, 2.0), (-1, 1)])
    # random_surface.create_random_surface(25)
    # random_surface.contour(1, 2, [1.0])
    # random_surface.graph(1, [1, 0])
    dig_range = (0, 1)
    off_diag_range = (-1, 1)
    print(RandomSurface.random_positive_definite_mat(1, dig_range, off_diag_range))
    print(RandomSurface.random_positive_definite_mat(2, dig_range, off_diag_range))
    print(RandomSurface.random_positive_definite_mat(3, dig_range, off_diag_range))