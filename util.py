import numpy as np


def scale(val, range):
    """ scales a given normalized input to span a certain range """
    return val * (range[1] - range[0]) + range[0]


def scale_vec(vec, ranges):
    vec = np.array(vec)
    if (vec.ndim == 1): return list(map(scale, vec, ranges))
    scaled = np.empty(vec.shape)
    for i in range(vec.shape[1]):
        scaled[:,i] = scale(vec[:,i], ranges[i])
    return scaled


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
    if off_diag_range[0] > 0:
        low = np.sqrt(off_diag_range[0]/dim)
        high = np.sqrt(off_diag_range[1]/dim)
    elif off_diag_range[1] < abs(off_diag_range[0]):
        low = - np.sqrt(off_diag_range[1]/dim)
        high = off_diag_range[0]/(dim*low)
    else:
        high = np.sqrt(off_diag_range[1] / dim)
        low = off_diag_range[0] / (dim * high)
    # ensure that Q is full rank (can be done more efficiently)
    while np.linalg.matrix_rank(Q) < dim:
        Q = scale(np.random.random([dim, dim]), (low, high))
    Q = Q.transpose().dot(Q); np.fill_diagonal(Q, 0)
    M = np.zeros([dim, dim])
    # ensure that M is positive definite (can be done more efficiently)
    while not np.all(np.linalg.eigvals(M) > 0):
        # TODO: raise a timeout exception when this loop executes repeatedly.
        M = Q + np.diag(scale(np.random.random(dim), diag_range))
    return M


def gaussian_kernel(x1, x2, g, W=None):
    if (x1.ndim == 1): x1 = x1[np.newaxis, :]
    if (x2.ndim == 1): x2 = x2[np.newaxis, :]
    if W is None:
        W = np.identity(x1.shape[1])
    else:
        W = np.diag(W)
    return np.exp(np.einsum('ij,ij->i', -g * (x2 - x1).dot(W), x2 - x1))