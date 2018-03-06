import numpy.random as rng
import numpy as np

wdecay = 0.0001

def create_dataset():
    """
    Creates a small dataset of 2d points in two linearly separable classes.
    :return: datapoints, labels
    """

    data_per_class = 12

    rng_state = rng.get_state()
    rng.seed(0)

    x1 = rng.multivariate_normal([-6, 0], np.eye(2), data_per_class)
    x2 = rng.multivariate_normal([+6, 0], np.eye(2), data_per_class)

    y1 = np.zeros(data_per_class)
    y2 = np.ones(data_per_class)

    xs = np.concatenate([x1, x2], axis=0)
    ys = np.concatenate([y1, y2], axis=0)

    rng.set_state(rng_state)

    return xs, ys
 
                 
def create_grid(xmin, xmax, N):
    """
    Creates a grid for 3d plotting.
    :param xmin: lower limit
    :param xmax: upper limit
    :param N: number of points in the grid per dimension
    :return: the grid
    """
    xx = np.linspace(xmin, xmax, N)
    X, Y = np.meshgrid(xx, xx)
    data = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)

    return data, X, Y
