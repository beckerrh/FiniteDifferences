import numpy as np
import scipy as sp
import time
import scipy.sparse.linalg as splinalg
from linalg import update

#=================================================================#
def getlgs(matrix, grid, d):
    ofs = matrix.offsets
    uofs = np.unique(np.abs(ofs))
    uofs = uofs[(uofs!=0)]
    strides = grid.strides()
    if sorted(uofs) != sorted(grid.strides()): raise KeyError(f"nonmatching uofs = {uofs} strides={grid.strides()}")
    # np.savetxt(f"lgs_{grid.nall()}_matrix.txt", matrix.toarray(), fmt="%6.2f")
    for uof in uofs:
        if uof != strides[d]: continue
        ind = 3*[0]
        for i in range(len(ofs)):
            if ofs[i] == 0: ind[0] = i
            elif ofs[i] == uof: ind[1] = i
            elif ofs[i] == -uof: ind[2] = i
        # print("uof", uof, "ind", ind)
        # print("matrix.data[ind]", matrix.data[ind])
        band = sp.sparse.dia_matrix((matrix.data[ind], ofs[ind]), shape=matrix.shape)
        # np.savetxt(f"lgs_{grid.nall()}_band_{uof}.txt", band.toarray(), fmt="%6.2f")
        lu = splinalg.splu(band.tocsc())
        return splinalg.LinearOperator(matrix.shape, lu.solve)

#=================================================================#
class MultiGrid:
    """
    finest mesh: level=0
    multigrid solver based on driver, which has to provide:
        - meshes()
        - matrices(grids)
        - newVector(grid)
        - newPreconditioner(matrix)
        - newCoarseSolver(matrix)
        - dirichletzero(grid, r)
        - restrict(l, f[l+1], v[l]) # from l+1 to l
        - prolongate(l, v[l], u[l+1]) # from l+1 to l
    """
    def __del__(self):
        if self.verbose:
            tot = sum(self.timer.values())
            if tot < 1e-3: return
            rel = {k:100*v/tot for k,v in self.timer.items()}
            tmin = int(tot/60)
            tsec = tot%60
            print(f"{self.__class__.__name__} finished\n total = {tmin:3d}[m] {tsec:4.1f}[s]")
            for k,v in rel.items(): print(f"{k:12s} {v:8.2f} %")
    def __init__(self, driver, **kwargs):
        self.grids, self.matrices = driver.getMeshesAndMatrices()
        if 'verbose' in kwargs: self.verbose=kwargs.pop('verbose')
        else: self.verbose=False
        assert len(self.grids) == len(self.matrices)
        self.driver = driver
        self.rtol, self.gtol, self.maxiter = 1e-10, 1e-14, 100
        self.minlevel, self.maxlevel, self.cycle = 0, self.driver.maxlevel, 1
        if 'maxiter' in kwargs: self.maxiter = self.maxiter = kwargs.pop('maxiter')
        self.f, self.u, self.v, self.update = [], [], [], []
        for grid in self.grids:
            self.f.append(driver.newVector(grid))
            self.u.append(driver.newVector(grid))
            self.v.append(driver.newVector(grid))
        self.prec = []
        for matrix, grid in zip(self.matrices, self.grids):
            self.prec.append(driver.newSmoothers(matrix,grid))
            self.update.append(update.Update(matrix))
        self.lu = driver.newCoarseSolver(self.matrices[self.maxlevel])
        self.timer = {'solve': 0, 'smooth':0, 'transfer': 0}
    def smoothpre(self, l, u, f, r): return self.smooth(l, u, f, r)
    def smoothpost(self, l, u, f, r): return self.smooth(l, u, f, r)
    def smoothcoarse(self, l, u, f, r):
        # return self.smooth(l, u, f, 1)
        return self.lu(f)
        # print(f"{l} id(f) {id(u)} {self.adress(u)}")
        # u = splinalg.spsolve(self.matrices[l], f)
        # print(f"{l} id(f) {id(u)} {self.adress(u)}")
        # return u
    def smoothold(self, l, u , f, r, maxiter):
        for iter in range(maxiter):
            r = self.residual(l, r, u, f)
            if np.linalg.norm(r) <= self.gtol: return u
            # print(f"\titer={iter}({l}) res={np.linalg.norm(r)}")
            u += self.prec[l].matvec(r)
        return u
    def smooth(self, l, u , f, r):
        for iter, prec in enumerate(self.prec[l]):
            if iter==0: r = self.residual(l, r, u, f)
            if np.linalg.norm(r) <= self.gtol: return u
            # print(f"\titer={iter}({l}) res={np.linalg.norm(r)} prec={prec}")
            # u += prec.matvec(r)
            self.update[l].update(u, prec.matvec(r), f, r)
        return u
    def residual(self, l, r, u , f):
        r[:] = f[:]
        r -=  self.matrices[l].dot(u)
        # r[np.abs(r)<np.finfo(np.float).eps]=0.0
        # self.driver.dirichletzero(self.grids[l], r)
        return r
    def update(self, l , u, v):
        u += v
        return u
    def solve(self, f, x0=None, verbose=True):
        self.f[0] = f
        if x0 is not None: self.u[0] = x0
        res0 = np.linalg.norm(f)
        tol = max(self.gtol, res0*self.rtol)
        resall = []
        for iter in range(1,self.maxiter+1):
            res = self.step(self.minlevel, self.u, self.f, self.v)
            resall.append(res)
            if iter>2 and res > resall[-2] and resall[-2] > resall[-3]:
                raise ArithmeticError(f"non monotone\nres={resall}\ngridf={self.grids[0]}\n nlev={len(self.grids)}")
            # if iter==1: res0 = res
            if verbose: print(f"mg({self.driver}) = {iter:3d} res={res:12.4e}")
            if res <= tol: return self.u[0], resall
        print(f"*** not converged*** res={res} (res0={res0})")
        return self.u[0], resall
    def step(self, l, u, f, v):
        if l == self.maxlevel:
            t0 = time.time()
            u[l] = self.smoothcoarse(l, u[l], f[l], v[l])
            self.timer['solve'] += time.time()-t0
            if l == self.minlevel:
                self.residual(l, v[l], u[l], f[l])
                return np.linalg.norm(v[l])
        else:
            t0 = time.time()
            self.smoothpre(l, u[l], f[l], v[l])
            self.timer['smooth'] += time.time()-t0
            self.residual(l, v[l], u[l], f[l])
            if l == self.minlevel: res = np.linalg.norm(v[l])
                # print(f"l = {l} res={res} f = {np.linalg.norm(f[l])} u = {np.linalg.norm(u[l])}")
            t0 = time.time()
            self.driver.restrict(self.grids[l], self.grids[l+1], f[l+1], v[l])
            self.timer['transfer'] += time.time()-t0
            self.driver.dirichletzero(self.grids[l+1], f[l+1])
            u[l+1].fill(0)
            for i in range(self.cycle):
                self.step(l+1, u, f, v)
            # v[l].fill(0)
            t0 = time.time()
            self.driver.prolongate(self.grids[l], self.grids[l+1], v[l], u[l+1])
            self.timer['transfer'] += time.time()-t0
            # self.driver.dirichletzero(self.grids[l], v[l])
            self.update[l].update(u[l], v[l], f[l])
            t0 = time.time()
            self.smoothpost(l, u[l], f[l], v[l])
            self.timer['smooth'] += time.time()-t0
            if l == self.minlevel: return res

#=================================================================#
class FdDriver:
    def __repr__(self):
        return f"{self.smoothers}"
    def __init__(self, gridf, A=None, **kwargs):
        from strucfem import matrix, grid
        if 'verbose' in kwargs: verbose=kwargs.pop('verbose')
        else: verbose=False
        n = gridf.n
        maxlevelgrids = self._getmaxlevelgrids(n)
        if 'maxlevel' in kwargs:
            maxlevel = kwargs.pop('maxlevel')
            if maxlevel >= 0: self.maxlevel = min(maxlevel, maxlevelgrids-1)
            else: self.maxlevel = min(self._getmaxlevelmg(-maxlevel, gridf.nall(), gridf.dim), maxlevelgrids-1)
        else:
            self.maxlevel = maxlevelgrids-1
        if verbose: print(f"maxlevelgrids = {maxlevelgrids} maxlevel = {self.maxlevel} nfine = {gridf.nall()}")
        self.grids, self.matrices = [], []
        self.grids.append(gridf)
        for l in range(self.maxlevel):
            n = (n-1)//2+1
            self.grids.append(grid.Grid(n=n, bounds=gridf.bounds))
        for i,grid in enumerate(self.grids):
            if i==0 and A is not None: self.matrices.append(A)
            else: self.matrices.append(matrix.createMatrixDiff(grid))
        if 'smoothers' in kwargs: self.smoothers=kwargs.pop('smoothers')
        else: raise KeyError(f"please give list of smoother")
        self.rhomat = np.abs(splinalg.eigs(self.matrices[-1], k=1)[0][0])
        if verbose: print(f"self.rhomat = {self.rhomat}")

    def _getmaxlevelgrids(self, n):
        import sympy
        maxlevel=10000
        for i in n:
            try:
                maxlevel= min(maxlevel, sympy.factorint(int(i-1))[2])
            except:
                maxlevel=1
        return maxlevel
    def _getmaxlevelmg(self, n0, nf, d):
        # 2^{d*l}nc = nf  & nc >= n0  --> log2(nf) >= d*l + log2(n0)
        difflog = np.log2(nf)-np.log2(max(2,n0))
        # print("difflog",difflog)
        if difflog<0: return 0
        return int( difflog /d +1)

    def _smooth_jac(self, matrix, grid, omega=0.8):
        D = omega/matrix.diagonal()
        return splinalg.LinearOperator(matrix.shape, lambda x: D*x)
    def _smooth_gs(self, matrix, grid, omega=0.8):
        lu= splinalg.splu(sp.sparse.tril(matrix).tocsc())
        return splinalg.LinearOperator(matrix.shape, lambda x: omega*lu.solve(x))
    def _smooth_ilu(self, matrix, grid, fill_factor=1, omega=0.1):
        d = omega*self.rhomat
        # d = 10*self.rhomat*h
        # d = 1100*self.rhomat*h*h
        # print(f"h = {h} d={d}")
        ilu = splinalg.spilu((matrix+d*sp.sparse.identity(matrix.shape[0])).tocsc(), fill_factor=fill_factor)
        return splinalg.LinearOperator(matrix.shape, ilu.solve)
    def _smooth_lgs(self, matrix, grid, omega=1):
        lus = []
        for d in range(grid.dim):
            lus.append(getlgs(matrix=matrix, grid=grid, d=d))
        return lus
    def getMeshesAndMatrices(self): return self.grids, self.matrices
    def newVector(self, grid): return np.zeros(grid.nall())
    def newSmoothers(self, matrix, grid):
        smoothers=[]
        for smoother in self.smoothers:
            try:
                smoothers.extend(self.newSmoother(smoother, matrix, grid))
            except:
                smoothers.append(self.newSmoother(smoother, matrix, grid))
        # print(f"smoothers = {len(smoothers)}")
        return smoothers
    def newSmoother(self, smoother, matrix, grid):
        if smoother == 'jac': return self._smooth_jac(matrix, grid)
        elif smoother == 'gs': return self._smooth_gs(matrix, grid)
        elif smoother == 'lgs': return self._smooth_lgs(matrix, grid)
        elif smoother == 'ilu': return self._smooth_ilu(matrix, grid)
        else: raise ValueError(f"unknown smoother {smoother}")
    def newCoarseSolver(self, matrix):
        return splinalg.factorized(matrix.tocsc())
    def dirichletzero(self, grid, r):
        dim = grid.dim
        assert r.shape[0] == grid.nall()
        r.shape = grid.n
        for i in range(dim):
            if grid.bdrycond[i][0] == 'dirichlet':
                np.moveaxis(r, i, 0)[0] = 0
            if grid.bdrycond[i][1] == 'dirichlet':
                np.moveaxis(r, i, 0)[-1] = 0
        r.shape = -1
        return r
    def restrict(self, gridf, gridc, v, u):
        from strucfem import transfer
        v.fill(0)
        return transfer.interpolate(gridf=gridf, gridc=gridc, uold=u, unew=v, transpose=True)
    def prolongate(self, gridf, gridc, v, u):
        from strucfem import transfer
        v.fill(0)
        return transfer.interpolate(gridf=gridf, gridc=gridc, unew=v, uold=u)


#=================================================================#
if __name__ == '__main__':
    import simfempy.tools.analyticalsolution as anasol
    from strucfem import matrix, grid, plotgrid
    import matplotlib.pyplot as plt

    d, l = 3, 6
    expr = ''
    for i in range(d): expr += f"log(1+x{i}**2)*"
    expr = expr[:-1]
    # expr = '3+x1+x0'
    uex = anasol.AnalyticalSolution(d, expr)
    n = np.array(d*[2**l+1])
    # n = 2**(l-2)*np.array([2,16,2])+1
    bounds = d*[[1,30]]
    bounds = 3*[[-1,1]]
    bounds[0] *= 100
    bounds[2] *= 10
    grid = grid.Grid(n=n, bounds=bounds)
    print(f"uex = {uex} n={grid.n} n={grid.nall()}")
    b = matrix.createRhsVectorDiff(grid, uex)
    smoothers = ['lgs', 'jac', 'ilu', 'gs']
    smoothers = ['lgs']
    # fd = FdDriver(grid, verbose=True, smoothers=smoothers, maxlevel=-10000)
    fd = FdDriver(grid, verbose=True, smoothers=smoothers)
    mg = MultiGrid(fd, verbose=True)
    u,res = mg.solve(b)
    # plotgrid.plot(grid, u=u)
    # plt.show()
