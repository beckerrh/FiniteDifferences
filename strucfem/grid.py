import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------------#
class Grid():
    def __repr__(self):
        return f" Grid(dim={self.dim}) n={self.n}, bounds={self.bounds} dx={self.dx} bdrycond={self.bdrycond}"
    def __init__(self, n, bounds):
        self.n = np.asarray(n)
        if len(self.n.shape)!=1:
            raise ValueError(f"Problem:  self.n.shape={self.n.shape}")
        self.dim = len(n)
        self.bounds = np.asarray(bounds)
        # self.bdrycond = self.dim*[[None,None]]
        self.bdrycond = self.dim*[['dirichlet','dirichlet']]
        if self.bounds.shape[0]!=self.dim:
            raise ValueError(f"Problem: dim={self.dim} self.bounds.shape={self.bounds.shape}")
        self.dx = np.empty(self.dim)
        for i in range(self.dim):
            # if self.n[i] % 2 != 1:
            #     raise ValueError(f"Problem: all n must be even numbers. Given n={n}")
            self.dx[i] = (self.bounds[i][1]-self.bounds[i][0])/float(self.n[i]-1)
    def nall(self):
        return np.prod(self.n)
    def volumeK(self):
        return np.prod(self.dx)
    def strides(self):
        strides = np.empty(self.dim, dtype=int)
        for i in range(self.dim):
            # print(f"i = {i} self.n[i+1:]={self.n[i+1:]}")
            strides[i] = int(np.prod(self.n[i+1:]))
            # strides[i] = int(np.prod(list(reversed(self.n))[i+1:]))
        return strides
    def x(self):
        x = []
        for i in range(self.dim):
            x.append(np.linspace(self.bounds[i][0], self.bounds[i][1], self.n[i]))
        return x
    def xc(self):
        x = []
        for i in range(self.dim):
            x.append(np.linspace(self.bounds[i][0]+0.5*self.dx[i], self.bounds[i][1]-0.5*self.dx[i], self.n[i]-1))
        return x
    def coord(self):
        return np.array(np.meshgrid(*self.x(), indexing='ij'))
    def coordCenters(self):
        return np.array(np.meshgrid(*self.xc(), indexing='ij'))

#=================================================================#
if __name__ == '__main__':
    from strucfem import plotgrid

    grid1 = Grid(n=[5], bounds=[[1,3]])
    grid2 = Grid(n=[5, 6], bounds=[[1,3], [2,4]])
    grid3 = Grid(n=[3, 4, 5], bounds=[[1,3], [2,4], [0,2]])
    # print(grid1, grid2, grid3)
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    plotgrid.plot(grid1, ax=ax, title="1D")
    ax = fig.add_subplot(3, 1, 2)
    plotgrid.plot(grid2, ax=ax, title="2D")
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    plotgrid.plot(grid3, ax=ax, title="3D")
    plt.show()

    # grid1.plot()
    # plt.show()
    # grid2.plot()
    # x,y = grid2.coord()
    # plt.plot(x[1,:], y[1,:], 'bo')
    # ufct = lambda x,y: x**2
    # u = ufct(x,y)
    # print("np.info(u)", np.info(u))
    # print("u[1,:]", u[1,:])
    # # u[1,:] = 10
    # print("u", u)
    # print("u.reshape(grid2.n[0], grid2.n[1])", u.reshape(grid2.n[0], grid2.n[1]))
    # cnt = plt.contourf(x, y, u)
    # plt.show()
    # grid3.plot()
    # plt.show()
    #
