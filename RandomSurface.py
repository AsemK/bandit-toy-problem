import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

class RandomSurface:
    """
    A simple class for creating random continuous functions in any dimension. The
    functions are created using multiples of gaussian multivariate distribution PDFs
    named "rocks".
    """
    def __init__(self, pos_ranges, values_range=(0.0, 1.0), cov_diag_range=(0.005, 0.01),
                 cov_off_diag_range=(-0.005, 0.005), smooth=True):
        """
        :param pos_ranges: the domain of the function along each dimension as a list of tuples.
        :param values_range: The range of the function
        :param cov_diag_range: The range of the diagonal elements of the randomly
        generated covariance matrices. This controls the range of the "width" of the
        gaussian shape along principle axes. (MUST BE POSITIVE)
        :param cov_off_diag_range: The range of the off diagonal elements of the
        randomly generated covariance matrices. This controls the rotation of the
        gaussian shape (MUST NOT BE LARGER THAN THE MAXIMUM VALUE OF cov_diag_range
        :param smooth: if true, graphing function will graph smooth surfaces,
        else rough surfaces will be graphed. Also self.get_surface_at will obey the
        chosen option
        """
        self.pos_ranges = pos_ranges
        self.values_range = values_range
        self.cov_diag_range = cov_diag_range
        self.cov_off_diag_range = cov_off_diag_range
        self.rocks = []
        self.smooth = smooth

    def get_dim(self):
        """ Returns the number of dimensions """
        return len(self.pos_ranges)

    def get_rocks_num(self):
        return len(self.rocks)

    @staticmethod
    def scale(val, low, high):
        """ scales a given normalized input to span a certain range """
        return val*(high-low)+low

    @staticmethod
    def random_positive_definite_mat(dim, diag_range, off_diag_range):
        """
        Generates a random positive definite matrix for any dimension to be used in
        generating random gaussian distributions.

        Note that any positive definite matrix can be decomposed to
        Q.transpose() * D * Q, where Q is any full rank matrix and D is a diagonal
        positive matrix.
        A slightly different way is used below to make it easier to force ranges on the
        generated matrix elements, which very important to effectively control the
        shape of the generated gaussian distributions.

        This function may fail (leading to an infinite loop) if the input off_diag_range
        allows values higher than those in the in the diag_range.
        TODO: raise exception if the input diag_range and off_diag_range are not valid
        """
        Q = np.zeros([dim, dim])
        high = np.sqrt(off_diag_range[1])
        low = high - np.sqrt(off_diag_range[1]-off_diag_range[0])
        # ensure that Q is full rank (can be done more efficiently)
        while np.linalg.matrix_rank(Q) < dim:
            Q = RandomSurface.scale(np.random.random([dim, dim]), low, high)
        Q = Q.transpose().dot(Q); np.fill_diagonal(Q, 0)
        M = np.zeros([dim, dim])
        # ensure that M is positive definite (can be done more efficiently)
        while not np.all(np.linalg.eigvals(M) > 0):
            # TODO: raise a timeout exception when this loop executes repeatedly.
            M = Q + np.diag(RandomSurface.scale(np.random.random(dim), *diag_range))
        return M

    def random_gaussian_surface(self):
        """
        creates a random gaussian distribution with the help of the Scipy funcion
        multivariate_normal
        :return: a tuple containing: rv: an object that allows extracting the PDF at
        certain position, as well as mean_pos and cov.
        """
        mean_pos = [self.scale(np.random.random(), *self.pos_ranges[i])
                    for i in range(self.get_dim())]
        cov = self.random_positive_definite_mat(self.get_dim(), self.cov_diag_range,
                                                self.cov_off_diag_range)
        rv = multivariate_normal(mean_pos, cov)
        return rv, mean_pos, cov

    def create_random_surface(self, rocks):
        """
        creates a random function by generating multiple randomly generated gaussian
        distributions.
        :param rocks: number of randomly generated gaussian distributions
        """
        heights = np.random.random(rocks)
        heights /= max(heights)
        self.rocks = []
        for i in range(rocks):
            self.rocks.append((*self.random_gaussian_surface(), heights[i]))

    def get_rough_surface_at(self, pos):
        """
        Returns the randomly generated function at the input position. The function is
        calculated as the maximum of all "rocks". This results in a normalized function
        that is piecewise continuous but not smooth. Here we call it a "rough" function.
        The number of local maxima of the rough function is less than
        or equal the number of rocks.
        :param pos: can be a single position or a vector of positions each has number
        of elements equal to the number of dimensions.
        :return: The randomly generated rough function valued at the input pos.
        """
        result = 0
        for rock in self.rocks:
            result = np.maximum(result, rock[0].pdf(pos)*rock[-1]/rock[0].pdf(rock[1]))
        result = self.scale(result, *self.values_range)
        return result

    def get_rough_highest_pos(self):
        """
        :return: The position of the global maximum of the rough function
        """
        max_ind = np.argmax([rock[-1] for rock in self.rocks])
        return self.rocks[max_ind][1]

    def get_smooth_surface_at(self, pos):
        """
        Returns the randomly generated function at the input position. The function is
        calculated as the sum of all "rocks". This results in a (nearly) normalized
        smooth function. The number of local maxima of the smooth function is less than
        or equal the number of rocks.
        :param pos: can be a single position or a vector of positions each has number
        of elements equal to the number of dimensions.
        :return: The randomly generated smooth function valued at the input pos.
        """
        # note that this is not necessarily the true maximum height of the surface.
        # so normalization is, in fact, not guaranteed.
        height = max(self.get_smooth_rocks_heights())
        result = 0
        for i, rock in enumerate(self.rocks):
            result += rock[0].pdf(pos) * rock[-1] / rock[0].pdf(rock[1]) / height
        result = self.scale(result, *self.values_range)
        return result

    def get_smooth_rocks_heights(self):
        """
        :return: the sum of all rocks valued at each rock mean.
        """
        heights = []
        for rock in self.rocks:
            heights += [sum(r[0].pdf(rock[1]) / r[0].pdf(r[1]) * r[-1]
                            for r in self.rocks)]
        return heights

    def get_smooth_highest_pos(self):
        """
        :return: The position of the rock mean that has the highest value in the smooth
        function. Note that This is not necessarily the position of the global maximum
        of the smooth function since the global maximum of the sum of two functions is
        not necessarily on the same position as the global maximum of one of them.
        However, In case the range cov_diag_range is small enough it can be shown that
        the returned position is very close to the position of the true global maximum.
        TODO: Implement a grid search to find the position of the real global maximum
        """
        max_ind = np.argmax(self.get_smooth_rocks_heights())
        return self.rocks[max_ind][1]

    def get_surface_at(self, pos, smooth=None):
        """
        A common interface for rough and smooth surfaces.
        :param pos: can be a single position or a vector of positions each has number
        of elements equal to the number of dimensions.
        :param smooth: a boolean to choose either a smooth or a rough surface. If not
        specified, the option chosen at initialization will be used.
        :return: The randomly generated  function valued at the input pos.
        """
        if not smooth: smooth = self.smooth
        if smooth:
            return self.get_smooth_surface_at(pos)
        else:
            return self.get_rough_surface_at(pos)

    def get_2D_highest_pos(self, axis1=0, axis2=1, other_axes_values=[],
                           precision=0.01, smooth=None):
        """
        :param smooth: a boolean to choose either a smooth or a rough surface. If not
        specified, the option chosen at initialization will be used.
        For description of the other parameters, see the documentation of the mesh_pos
        method.
        :return: the position of the highest value on 2D data.
        """
        x, y, pos = self.mesh_pos(axis1, axis2, other_axes_values, precision)
        result = self.get_surface_at(pos, smooth)
        max_ind = np.unravel_index(np.argmax(result), result.shape)
        return [x[max_ind], y[max_ind]]

    def graph(self, axis=0, other_axes_values=[], precision=0.01, smooth=None):
        """
        produces 2D graph of one axis of the generated function with all others fixed
        at certain values
        :param axis: graphed axis number
        :param other_axes_values: fixed values for other axes as an ordered list,
        if not provided, all axes will be fixed at zero.
        :param precision: resolution of the generated graph.
        :param smooth: a boolean to choose either a smooth or a rough surface. If not
        specified, the option chosen at initialization will be used.
        """
        if self.get_dim() < 1 or self.get_dim() < axis:
            # TODO: raise an exception
            return

        # calculate position vector
        x = np.arange(*self.pos_ranges[axis], precision)
        pos = np.zeros((len(x), self.get_dim()))
        c = 0
        for i in range(self.get_dim()):
            if i == axis:
                pos[:, axis] = x
            elif other_axes_values:
                pos[:, i] = other_axes_values[c]*np.ones(x.shape)
                c += 1

        # plot
        plt.figure()
        plt.plot(x, self.get_surface_at(pos, smooth))
        plt.show()

    def mesh_pos(self, axis1, axis2, other_axes_values=[], precision=0.01):
        """
        calculates a position mesh grid for contour or 3d graphing
        :param axis1: number of axis to be placed in the x-axis of the graph
        :param axis2: number of axis to be placed in the y-axis of the graph
        :param other_axes_values: fixed values of other axes (in case the generated
        function has a dimension higher than 2. if not provided, all other axes will be
        fixed at zero
        :param precision: resolution of the graph.
        """
        if self.get_dim() < 2 or self.get_dim() < axis1 or self.get_dim() < axis2:
            # TODO: raise an exception
            return

        x, y = np.meshgrid(np.arange(*self.pos_ranges[axis1], precision),
                           np.arange(*self.pos_ranges[axis2], precision))
        pos = np.zeros(x.shape + (self.get_dim(),))
        c = 0
        for i in range(self.get_dim()):
            if i == axis1:
                pos[:, :, i] = x
            elif i == axis2:
                pos[:, :, i] = y
            elif other_axes_values:
                pos[:, :, i] = other_axes_values[c] * np.ones(x.shape)
                c += 1
        return x, y, pos

    def contour(self, axis1=0, axis2=1, other_axes_values=[], precision=0.01, smooth=None):
        """
        Generates a 2D contour graph of the generated function for two axes with the
        rest fixed.
        :param smooth: a boolean to choose either a smooth or a rough surface. If not
        specified, the option chosen at initialization will be used.
        For description of the other parameters, see the documentation of the mesh_pos
        method.
        """
        x, y, pos = self.mesh_pos(axis1, axis2, other_axes_values, precision)

        plt.figure()
        plt.contourf(x, y, self.get_surface_at(pos, smooth), levels=np.arange(
            self.values_range[0], self.values_range[1]+0.2, 0.05), cmap=cm.Reds)
        plt.axis('square')
        plt.axis([*self.pos_ranges[axis1], *self.pos_ranges[axis2]])
        plt.show()

    def graph3d(self, axis1=0, axis2=1, other_axes_values=[], precision=0.01, smooth=None):
        """
        Generates a 3D graph of the generated function for two axes with the rest fixed.
        :param smooth: a boolean to choose either a smooth or a rough surface. If not
        specified, the option chosen at initialization will be used.
        For description of the other parameters, see the documentation of the mesh_pos
        method.
        """
        x, y, pos = self.mesh_pos(axis1, axis2, other_axes_values, precision)
        plt.figure()
        plt.gca(projection='3d').plot_surface(x, y, self.get_surface_at(pos, smooth), color='red')
        plt.axis('square')
        plt.show()


if __name__ == '__main__':
    random_surface = RandomSurface([(0, 1.0), (0, 1.0)], smooth=False)
    random_surface.create_random_surface(30)
    print(random_surface.get_rough_surface_at(random_surface.get_rough_highest_pos()))
    print(random_surface.get_rough_highest_pos())
    print(random_surface.get_smooth_surface_at(random_surface.get_smooth_highest_pos()))
    print(random_surface.get_smooth_highest_pos())
    random_surface.contour()
    random_surface.graph3d()

    # diag_range = (0.1, 0.2)
    # off_diag_range = (-0.1, 0.1)
    # print(RandomSurface.random_positive_definite_mat(1, diag_range, off_diag_range))
    # print(RandomSurface.random_positive_definite_mat(2, diag_range, off_diag_range))
    # print(RandomSurface.random_positive_definite_mat(3, diag_range, off_diag_range))