import numpy as np
import scipy.sparse.linalg as splinalg
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, transfer, plotgrid
from linalg import mg

#
# def dirichletzero(grid, u):
#     # print(f"dirichletzero {id(u)}")
#     dim = grid.dim
#     assert u.shape[0] == grid.nall()
#     u.shape = grid.n
#     for i in range(dim):
#         if grid.bdrycond[i][0] == 'dirichlet':
#             np.moveaxis(u, i, 0)[0] = 0
#         if grid.bdrycond[i][1] == 'dirichlet':
#             np.moveaxis(u, i, 0)[-1] = 0
#     u.shape = -1
#     # u = u.ravel()
#     # print(f"dirichletzero {id(u)}")
#     return u
#
# #=================================================================#
# class MultiGrid:
#     def __init__(self, grids, matrices, maxlevel=None):
#         self.rtol, self.gtol, self.maxiter = 1e-12, 1e-15, 100
#         assert len(grids) == len(matrices)
#         self.grids = grids
#         self.matrices = matrices
#         self.minlevel, self.maxlevel, self.cycle = 0, len(grids)-1, 1
#         self.omega = 0.8
#         self.maxiterpre, self.maxiterpost = 2,2
#         if maxlevel is not None: self.maxlevel = min(maxlevel,len(grids)-1)
#         for grid in grids:
#             print(f"grid = {grid}")
#         for matrix in matrices:
#             print(f"matrix = {matrix.shape}")
#         self.f, self.u, self.v = [], [], []
#         for grid in grids:
#             self.f.append(np.zeros(grid.nall()))
#             self.u.append(np.zeros(grid.nall()))
#             self.v.append(np.zeros(grid.nall()))
#         self.Dinv = []
#         for matrix in matrices:
#             D = matrix.diagonal()
#             self.Dinv.append(1/D)
#         self.lu = splinalg.factorized(matrices[self.maxlevel])
#         # print(self.Dinv)
#     def adress(self, u):
#         return u.__array_interface__['data'][0]
#     def smoothpre(self, l, u, f, r): return self.smooth(l, u, f, r, self.maxiterpre)
#     def smoothpost(self, l, u, f, r): return self.smooth(l, u, f, r, self.maxiterpost)
#     def smoothcoarse(self, l, u, f, r):
#         # return self.smooth(l, u, f, 1)
#         return self.lu(f)
#         # print(f"{l} id(f) {id(u)} {self.adress(u)}")
#         # u = splinalg.spsolve(self.matrices[l], f)
#         # print(f"{l} id(f) {id(u)} {self.adress(u)}")
#         # return u
#     def smooth(self, l, u , f, r, maxiter):
#         for iter in range(maxiter):
#             r = self.residual(l, r, u, f)
#             # print(f"\titer={iter}({l}) res={np.linalg.norm(r)}")
#             u += self.omega*self.Dinv[l]*r
#         return u
#     def residual(self, l, r, u , f):
#         r[:] = f[:]
#         r -=  self.matrices[l].dot(u)
#         dirichletzero(self.grids[l], r)
#         return r
#     def addUpdate(self, l , u, v):
#         u += v
#         return u
#     def prolongate(self, l, v, u):
#         # print(f"prolongate {l+1} ===> {l}")
#         return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l + 1], unew=v, uold=u)
#     def restrict(self, l, v, u):
#         # print(f"restrict {l} ===> {l+1}")
#         v.fill(0)
#         return transfer.interpolate(gridf=self.grids[l], gridc=self.grids[l + 1], uold=u, unew=v, transpose=True)
#     def solve(self, u, f):
#         self.f[0] = f
#         self.u[0] = u
#         # for f,u in zip(self.f, self.u):
#         #     print(f"f={f.shape} u={u.shape}")
#         # plotgrid.plot(self.grids[0], u=self.u[0], title=f"iter = 0")
#         ids = [id(f) for f in self.v]
#         for iter in range(1,self.maxiter+1):
#             res = self.step(0, self.u, self.f, self.v)
#             if iter==1: res1 = res
#             for idf,f in zip(ids,self.v):
#                 if idf!=id(f): raise ValueError(f"id changed {idf} != {id(f)}")
#             print(f"mgiter = {iter:3d} res={res:12.4e}")
#             if res <= self.gtol or res <= res1*self.rtol:
#                 return self.u[0]
#             # plotgrid.plot(self.grids[0], u=self.u[0], title=f"iter = {iter}")
#             if res < self.gtol: return self.u[0]
#     def step(self, l, u, f, v):
#         # print(f"level = {l} maxlevel = {self.maxlevel}")
#         # print(f"u=\n{self.u}\nf=\n{self.f}\nv=\n{self.v}")
#         if l == self.maxlevel:
#             # print(f"{l} id(f) {id(u[l])} {self.adress(u[l])}")
#             u[l] = self.smoothcoarse(l, u[l], f[l], v[l])
#             # print(f"{l} id(f) {id(u[l])} {self.adress(u[l])}")
#             if l == self.minlevel:
#                 v[l].fill(0)
#                 self.residual(l, v[l], u[l], f[l])
#                 return np.linalg.norm(v[l])
#         else:
#             u[l] = self.smoothpre(l, u[l], f[l], v[l])
#             v[l] = self.residual(l, v[l], u[l], f[l])
#             # print(f"1 {l} id {id(v[l])} {self.adress(v[l])}")
#             # print(f"restrict: {v[l]} --> {f[l+1]}")
#             f[l+1] = self.restrict(l, f[l+1], v[l])
#             dirichletzero(self.grids[l+1], f[l+1])
#             # print(f"restrict: {v[l]} --> {f[l+1]}")
#             u[l+1].fill(0)
#             for i in range(self.cycle):
#                 self.step(l+1, u, f, v)
#             v[l].fill(0)
#             self.prolongate(l, v[l], u[l+1])
#             dirichletzero(self.grids[l], v[l])
#             self.addUpdate(l,u[l], v[l])
#             # print(f"u = {u[l]}")
#             self.smoothpost(l, u[l], f[l], v[l])
#             if l == self.minlevel: return np.linalg.norm(v[l])

#=================================================================#
class MgDriver(mg.FdDriver):
    def __init__(self, grid):
        super().__init__(grid)

#=================================================================#
def solve(A, b, grid, x0=None):
    mgD = MgDriver(grid)
    mgS = mg.MultiGrid(mgD)
    return mgS.solve(b, x0=x0)

#=================================================================#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d, l = 2, 2
    expr = ''
    # for i in range(d): expr += f"log(1+x{i}**2)*"
    for i in range(d): expr += f"x{i}**2+"
    expr = expr[:-1]
    uex = anasol.AnalyticalSolution(d, expr)
    bounds=d*[[1,3]]
    grid = grid.Grid(n=np.array(d*[2**l+1]), bounds=bounds)
    b = matrix.createRhsVectorDiff(grid, uex)
    A = matrix.createMatrixDiff(grid)
    u, res = solve(A, b, grid)
    plotgrid.plot(grid, u=u)
    plt.show()
