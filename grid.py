import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#-----------------------------------------------------------------#
class Grid():
    def __repr__(self):
        return f" Grid({self.dim}) n={self.n}, length={self.length} dx={self.dx} bdrycond={self.bdrycond}"
    def __init__(self, n, length):
        self.n = n
        self.dim = len(n)
        self.length = np.asarray(length)
        self.bdrycond = self.dim*[[None,None]]
        if self.length.shape[0]!=self.dim:
            raise ValueError(f"Problem: dim={self.dim} self.length.shape={self.length.shape}")
        self.dx = np.empty(self.dim)
        for i in range(self.dim):
            if n[i] % 2 != 1:
                raise ValueError(f"Problem: all n must be even numbers. Given n={n}")
            self.dx[i] = (self.length[i][1]-self.length[i][0])/float(self.n[i]-1)
    def x(self):
        if self.dim == 1:
            return np.linspace(self.length[0][0], self.length[0][1], self.n[0])
        elif self.dim == 2:
            x = np.linspace(self.length[0][0], self.length[0][1], self.n[0])
            y = np.linspace(self.length[1][0], self.length[1][1], self.n[1])
            return np.meshgrid(x, y)
        elif self.dim == 3:
            x = np.linspace(self.length[0][0], self.length[0][1], self.n[0])
            y = np.linspace(self.length[1][0], self.length[1][1], self.n[1])
            z = np.linspace(self.length[2][0], self.length[2][1], self.n[2])
            return np.meshgrid(x, y, z)
        else:
            raise ValueError(f"Problem: unable to plot in d={self.dim}")

    def plot(self, ax = None):
        if self.dim==1:
            x = self.x()
            y = np.full_like(x, 1)
            if ax == None: ax = plt.gca()
            ax.plot(x, y, 'x-')
        elif self.dim == 2:
            xx, yy = self.x()
            if ax == None: ax = plt.gca()
            ax.plot(xx, yy, marker='x', color='r', linestyle='none')
        elif self.dim == 3:
            xx, yy, zz = self.x()
            if ax == None: ax = plt.gca(projection='3d')
            # if ax == None:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xx, yy, zz, color='r')
        else:
            raise ValueError(f"Problem: unable to plot in d={self.dim}")

#=================================================================#
if __name__ == '__main__':
    grid1 = Grid(n=[5], length=[[1,3]])
    grid2 = Grid(n=[5, 7], length=[[1,3], [2,4]])
    grid3 = Grid(n=[5, 7, 3], length=[[1,3], [2,4], [0,2]])
    print(grid1, grid2, grid3)

    # grid1.plot()
    # plt.show()
    # grid2.plot()
    # plt.show()
    # grid3.plot()
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    grid1.plot(ax)
    ax = fig.add_subplot(3, 1, 2)
    grid2.plot(ax)
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    grid3.plot(ax)
    plt.show()