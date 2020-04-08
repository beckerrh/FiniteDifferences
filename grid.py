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
        x = []
        for i in range(self.dim):
            x.append(np.linspace(self.length[i][0], self.length[i][1], self.n[i]))
        return x
    def coord(self):
        return np.meshgrid(*self.x(), indexing='ij')

    def plot(self, ax = None):
        if self.dim==1:
            x = self.coord()
            y = np.full_like(x, 1)
            if ax == None: ax = plt.gca()
            ax.plot(x, y, 'rx-')
        elif self.dim == 2:
            xx, yy = self.coord()
            if ax == None: ax = plt.gca()
            ax.plot(xx, yy, marker='x', color='r', linestyle='none')
        elif self.dim == 3:
            xx, yy, zz = self.coord()
            if ax == None: ax = plt.gca(projection='3d')
            ax.scatter(xx, yy, zz, color='r')
        else:
            raise ValueError(f"Problem: unable to plot in d={self.dim}")

#=================================================================#
if __name__ == '__main__':
    grid1 = Grid(n=[5], length=[[1,3]])
    grid2 = Grid(n=[5, 7], length=[[1,3], [2,4]])
    grid3 = Grid(n=[5, 7, 3], length=[[1,3], [2,4], [0,2]])
    # print(grid1, grid2, grid3)

    # grid1.plot()
    # plt.show()
    grid2.plot()
    x,y = grid2.coord()
    plt.plot(x[1,:], y[1,:], 'bo')
    ufct = lambda x,y: x**2
    u = ufct(x,y)
    print("np.info(u)", np.info(u))
    print("u[1,:]", u[1,:])
    # u[1,:] = 10
    print("u", u)
    print("u.reshape(grid2.n[0], grid2.n[1])", u.reshape(grid2.n[0], grid2.n[1]))
    cnt = plt.contourf(x, y, u)
    plt.show()
    # grid3.plot()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(3, 1, 1)
    # grid1.plot(ax)
    # ax = fig.add_subplot(3, 1, 2)
    # grid2.plot(ax)
    # ax = fig.add_subplot(3, 1, 3, projection='3d')
    # grid3.plot(ax)
    # plt.show()