import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

class RandomSurface:
    def __init__(self, pos_ranges, values_range=(0.0, 1.0), cov_diag_range=(0, 0.01), cov_off_diag_range=(-0.01, 0.01)):
        self.pos_ranges = pos_ranges
        self.values_range = values_range
        self.cov_diag_range = cov_diag_range
        self.cov_off_diag_range = cov_off_diag_range

        self.rocks = []
        self.get_surface_at = self.get_continuous_surface_at

    def get_dim(self):
        return len(self.pos_ranges)

    def get_rocks_num(self):
        return len(self.rocks)

    def get_values_range(self):
        return self.values_range

    @staticmethod
    def random_positive_definite_mat(dim, cov_diag_range, cov_off_diag_range):
        cov = np.random.random([dim, dim]) * (cov_off_diag_range[1]-cov_off_diag_range[0])+cov_off_diag_range[0]
        cov = (cov + cov.transpose())/2
        while not np.all(np.linalg.eigvals(cov) > 0):
            cov = (np.diag(np.random.random(dim))*(cov_diag_range[1]-cov_diag_range[0])+cov_diag_range[0]) \
                  + np.tril(cov, -1) + np.tril(cov, -1).transpose()
        return cov

    def random_gaussian_surface(self):
        mean_pos = [np.random.random()*(self.pos_ranges[i][1]-self.pos_ranges[i][0])+self.pos_ranges[i][0]
                    for i in range(self.get_dim())]
        cov = self.random_positive_definite_mat(self.get_dim(), self.cov_diag_range, self.cov_off_diag_range)
        rv = multivariate_normal(mean_pos, cov)
        return rv, mean_pos, cov

    def create_random_rough_surface(self, rocks):
        heights = np.random.random(rocks) * (self.values_range[1] - self.values_range[0]) + self.values_range[0]
        heights *= self.values_range[1]/max(heights)
        self.rocks = []
        for i in range(rocks):
            self.rocks.append((*self.random_gaussian_surface(), heights[i]))

    def get_continuous_surface_at(self, pos):
        result = np.zeros(len(pos))
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

        x = np.mgrid[self.pos_ranges[axis][0]:self.pos_ranges[axis][1]:precision]
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

        x, y = np.mgrid[self.pos_ranges[axis1][0]:self.pos_ranges[axis1][1]:precision,
                        self.pos_ranges[axis2][0]:self.pos_ranges[axis2][1]:precision]
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
        plt.contourf(x, y, self.get_surface_at(pos), levels = np.arange(*self.get_values_range(), 0.1))
        plt.show()


if __name__ == '__main__':
    random_surface = RandomSurface([(0.0, 1.0), (0.0, 1.0)])
    random_surface.create_random_rough_surface(25)
    random_surface.contour(0, 1, [])