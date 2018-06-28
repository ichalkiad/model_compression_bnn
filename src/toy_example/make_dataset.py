import numpy as np
import math
import matplotlib.pyplot as plt
import random

def create_dataset(xmin,xmax,N):

    A0 = 1.0
       
    T1 = 3
    phi1 = 2*math.pi/float(T1)
    A1 = 1/phi1
    
    T2 = 6
    phi2 = 2*math.pi/float(T2)
    A2 = 1/phi2
    
    xt = np.linspace(xmin, xmax, N)
    yt = np.zeros(xt.shape)
    for i in xrange(N):
        yt[i] = A0 + A1*math.sin(phi1*xt[i]) + A2*math.cos(phi2*xt[i])

    samples = np.random.normal(1,0.001,N/2)
    idx = random.sample(range(N), N/2)

    xs = np.zeros((N,2))
    xs[0:N/2,1] = yt[idx]+samples 
    xs[0:N/2,0] = xt[idx]
    nidx = np.setdiff1d(np.arange(0,N,1),idx)
    xs[N/2:N,1] = yt[nidx]-samples 
    xs[N/2:N,0] = xt[nidx]

    ys = np.zeros((N,))
    ys[0:N/2] = 1

    idx = random.sample(range(N), N/20)
    x = xs[idx,:]
    y = ys[idx]
    
    return x, y, xt, yt


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



def show_train_data():
    """
    Plots the training data.
    """
    xmin = -10.0
    xmax = 10.0
    N = 1000
    xs, ys, xt, yt = create_dataset(xmin,xmax,N)

    plt.figure()
    plt.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    plt.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    plt.plot(xt,yt)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.title('Training data')

    plt.show()


if __name__ == "__main__":
    show_train_data()

