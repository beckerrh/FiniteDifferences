import numpy as np
import scipy.sparse.linalg as splinalg
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, transfer, plotgrid


def dirichletzero(grid, u):
    dim = grid.dim
    assert u.shape[0] == grid.nall()
    u = u.reshape(grid.n)
    for i in range(dim):
        if grid.bdrycond[i][0] == 'dirichlet':
            np.moveaxis(u, i, 0)[0] = 0
        if grid.bdrycond[i][1] == 'dirichlet':
            np.moveaxis(u, i, 0)[-1] = 0
    return u.ravel()

#=================================================================#
class MultiGrid:
    def __init__(self, grids, matrices):
        self.rtol, self.gtol, self.maxiter = 1e-10, 1e-14, 10
        assert len(grids) == len(matrices)
        self.grids = grids
        self.matrices = matrices
        self.minlevel, self.maxlevel, self.cycle = 0, len(grids)-1, 1
        self.omega = 0.6
        self.maxiterpre, self.maxiterpost = 2,2
        for grid in grids:
            print(f"grid = {grid}")
        for matrix in matrices:
            print(f"matrix = {matrix.shape}")
        self.f, self.u, self.v = [], [], []
        for grid in grids:
            self.f.append(np.zeros(grid.nall()))
            self.u.append(np.zeros(grid.nall()))
            self.v.append(np.zeros(grid.nall()))
        self.Dinv = []
        for matrix in matrices:
            D = matrix.diagonal()
            self.Dinv.append(1/D)
        # print(self.Dinv)

    def smoothpre(self, l, u, f): return self.smooth(l, u, f, self.maxiterpre)
    def smoothpost(self, l, u, f): return self.smooth(l, u, f, self.maxiterpost)
    def smoothcoarse(self, l, u, f):
        # return self.smooth(l, u, f, 1)
        return  splinalg.spsolve(self.matrices[l],f)
    def smooth(self, l, u , f, maxiter):
        for iter in range(maxiter):
            r = self.residual(l, u, f)
            # print(f"\titer={iter}({l}) res={np.linalg.norm(r)}")
            u += self.omega*self.Dinv[l]*r
        return u
    def residual(self, l, u , f):
        r = f - self.matrices[l].dot(u)
        dirichletzero(self.grids[l], r)
        return r
    def addUpdate(self, l , u, v):
        u += v
        return u
    def prolongate(self, l, u):
        # print(f"prolongate {l+1} ===> {l}")
        return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l + 1], uold=u)
    def restrict(self, l, u):
        # print(f"restrict {l} ===> {l+1}")
        return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l + 1], uold=u, transpose=True)
    def solve(self, u, f, maxlevel=None):
        if maxlevel is not None: self.maxlevel=maxlevel
        self.f[0] = f
        self.u[0] = u
        # for f,u in zip(self.f, self.u):
        #     print(f"f={f.shape} u={u.shape}")
        plotgrid.plot(self.grids[0], u=self.u[0], title=f"iter = 0")
        for iter in range(1,self.maxiter+1):
            res = self.step(0, self.u, self.f, self.v)
            print(f"mgiter = {iter:3d} res={res:12.4e}")
            plotgrid.plot(self.grids[0], u=self.u[0], title=f"iter = {iter}")
            if res < self.gtol: return self.u[0]
    def step(self, l, u, f, v):
        print(f"level = {l} maxlevel = {self.maxlevel}")
        # print(f"u=\n{self.u}\nf=\n{self.f}\nv=\n{self.v}")
        if l == self.maxlevel:
            u[l] = self.smoothcoarse(l, u[l], f[l])
            if l == self.minlevel:
                v[l].fill(0)
                v[l] = self.residual(l, u[l], f[l])
                return np.linalg.norm(v[l])
        else:
            u[l] = self.smoothpre(l, u[l], f[l])
            v[l] = self.residual(l, u[l], f[l])
            # print(f"restrict: {v[l]} --> {f[l+1]}")
            f[l+1] = self.restrict(l, v[l])
            f[l+1] = dirichletzero(self.grids[l+1], f[l+1])
            # print(f"restrict: {v[l]} --> {f[l+1]}")
            u[l+1].fill(0)
            for i in range(self.cycle):
                self.step(l+1, u, f, v)
            v[l].fill(0)
            v[l] = self.prolongate(l, u[l+1])
            v[l] = dirichletzero(self.grids[l], v[l])
            u[l] = self.addUpdate(l,u[l], v[l])
            # print(f"u = {u[l]}")
            u[l] = self.smoothpost(l, u[l], f[l])
            if l == self.minlevel: return np.linalg.norm(v[l])

#=================================================================#
if __name__ == '__main__':
    d = 2
    ns = [np.array(d*[3])]
    for k in range(2): ns.append(2 * ns[k] - 1)
    expr = ''
    for i in range(d): expr += f"(1-x{i}**2)*"
    expr = expr[:-1]
    bounds=d*[[-1,1]]
    grids, matrices = [], []
    for n in reversed(ns):
        grids.append(grid.Grid(n=n, bounds=bounds))
        matrices.append(matrix.createMatrixDiff(grids[-1]))
    mg = MultiGrid(grids, matrices)
    u = anasol.AnalyticalSolution(grids[0].dim, expr)
    b = matrix.createRhsVectorDiff(grids[0], u)
    uh = np.zeros_like(b)
    uh = matrix.dirichlet(grids[0], uh, u)
    # print(f"uh = {uh}")
    mg.solve(uh, b, maxlevel=1)
