import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
import simfempy.tools.analyticalsolution as anasol
import grid, matrix, transfer

#=================================================================#
class MultiGrid:
    def __init__(self, grids, matrices):
        self.rtol, self.gtol, self.maxiter = 1e-10, 1e-14, 4
        assert len(grids) == len(matrices)
        self.grids = grids
        self.matrices = matrices
        self.minlevel, self.maxlevel, self.cycle = 0, len(grids)-1, 1
        self.omega = 0.6
        self.maxiterpre, self.maxiterpost = 0,4
        for grid in grids:
            print(f"grid = {grid}")
        for matrix in matrices:
            print(f"matrix = {matrix.shape}")
        self.f, self.u, self.v = [], [], []
        for grid in grids:
            self.f.append(np.empty(grid.nall()))
            self.u.append(np.empty(grid.nall()))
            self.v.append(np.empty(grid.nall()))
        self.Dinv = []
        for matrix in matrices:
            D = matrix.diagonal()
            self.Dinv.append(1/D)
        # print(self.Dinv)

    def smoothpre(self, l, u, f): return self.smooth(l, u, f, self.maxiterpre)
    def smoothpost(self, l, u, f): return self.smooth(l, u, f, self.maxiterpost)
    def smoothcoarse(self, l, u, f):
        return self.smooth(l, u, f, 10)
        return  splinalg.spsolve(self.matrices[l],f)
    def smooth(self, l, u , f, maxiter):
        for iter in range(maxiter):
            r = self.residual(l, u, f)
            print(f"\titer={iter} res={np.linalg.norm(r)}")
            u += self.omega*self.Dinv[l]*r
        return u
    def residual(self, l, u , f):
        return f - self.matrices[l].dot(u)
    def addUpdate(self, l , u, v):
        u += v
        return u
    def prolongate(self, l, u):
        print(f"prolongate {l+1} ===> {l}")
        return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l+1], uold=u[l+1])
    def restrict(self, l, u):
        print(f"restrict {l} ===> {l+1}")
        return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l+1], uold=u[l], transpose=True)
    def solve(self, f, maxlevel=None):
        if maxlevel is not None: self.maxlevel=maxlevel
        self.f[0] = f
        # for f,u in zip(self.f, self.u):
        #     print(f"f={f.shape} u={u.shape}")
        for iter in range(self.maxiter):
            res = self.step(0, self.u, self.f, self.v)
            print(f"iter = {iter:3d} res={res:12.4e}")
            if res < self.gtol: return self.u[0]
    def step(self, l, u, f, v):
        print(f"level = {l} maxlevel = {self.maxlevel}")
        if l == self.maxlevel:
            u[l] = self.smoothcoarse(l, u[l], f[l])
            if l == self.minlevel:
                v[l].fill(0)
                v[l] = self.residual(l, u[l], f[l])
                return np.linalg.norm(v[l])
        else:
            u[l] = self.smoothpre(l, u[l], f[l])
            v[l] = self.residual(l, u[l], f[l])
            f[l+1] = self.restrict(l, v)
            u[l+1].fill(0)
            for i in range(self.cycle):
                self.step(l+1, u, f, v)
            v[l].fill(0)
            v[l] = self.prolongate(l, u)
            u[l] = self.addUpdate(l,u[l], v[l])
            u[l] = self.smoothpost(l, u[l], f[l])
            if l == self.minlevel: return np.linalg.norm(v[l])

#=================================================================#
if __name__ == '__main__':
    d = 2
    ns = [np.array([3, 3])]
    for k in range(1): ns.append(2 * ns[k] - 1)
    expr = 'cos(pi*x)*cos(pi*y)'
    bounds=d*[[-1,1]]
    grids, matrices = [], []
    for n in reversed(ns):
        grids.append(grid.Grid(n=n, bounds=bounds))
        matrices.append(matrix.createMatrixDiff(grids[-1]))
    mg = MultiGrid(grids, matrices)
    u = anasol.AnalyticalSolution(grids[0].dim, expr)
    b = matrix.createRhsVectorDiff(grids[0], u)
    mg.solve(b, maxlevel=0)
