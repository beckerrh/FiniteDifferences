import numpy as np
import scipy.sparse.linalg as splinalg

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
    def __init__(self, driver, **kwargs):
        self.grids, self.matrices = driver.getMeshesAndMatrices()
        assert len(self.grids) == len(self.matrices)
        self.driver = driver
        self.rtol, self.gtol, self.maxiter = 1e-12, 1e-15, 100
        self.minlevel, self.maxlevel, self.cycle = 0, len(self.grids)-1, 1
        self.maxiterpre, self.maxiterpost = 2,2
        if 'maxlevel' in kwargs:
            self.maxlevel = min(kwargs.pop('maxlevel'),self.maxlevel)
        self.f, self.u, self.v = [], [], []
        for grid in self.grids:
            self.f.append(driver.newVector(grid))
            self.u.append(driver.newVector(grid))
            self.v.append(driver.newVector(grid))
        self.prec = []
        for matrix in self.matrices:
            self.prec.append(driver.newPreconditioner(matrix))
        self.lu = driver.newCoarseSolver(self.matrices[self.maxlevel])
    def smoothpre(self, l, u, f, r): return self.smooth(l, u, f, r, self.maxiterpre)
    def smoothpost(self, l, u, f, r): return self.smooth(l, u, f, r, self.maxiterpost)
    def smoothcoarse(self, l, u, f, r):
        # return self.smooth(l, u, f, 1)
        return self.lu(f)
        # print(f"{l} id(f) {id(u)} {self.adress(u)}")
        # u = splinalg.spsolve(self.matrices[l], f)
        # print(f"{l} id(f) {id(u)} {self.adress(u)}")
        # return u
    def smooth(self, l, u , f, r, maxiter):
        for iter in range(maxiter):
            r = self.residual(l, r, u, f)
            # print(help(self.prec[l]))
            # print(f"\titer={iter}({l}) res={np.linalg.norm(r)}")
            u += self.prec[l].matvec(r)
        return u
    def residual(self, l, r, u , f):
        r[:] = f[:]
        r -=  self.matrices[l].dot(u)
        return r
    def update(self, l , u, v):
        u += v
        return u
    def solve(self, f, x0=None, verbose=True):
        self.f[0] = f
        if x0 is not None: self.u[0] = x0
        res1 = np.linalg.norm(f)
        tol = max(self.gtol, res1*self.rtol)
        resall = []
        for iter in range(1,self.maxiter+1):
            res = self.step(self.minlevel, self.u, self.f, self.v)
            resall.append(res)
            # if iter==1: res1 = res
            if verbose: print(f"mgiter = {iter:3d} res={res:12.4e}")
            if res <= tol:
                return self.u[0], resall
        print(f"*** not converged***")
        return self.u[0], resall
    def step(self, l, u, f, v):
        if l == self.maxlevel:
            u[l] = self.smoothcoarse(l, u[l], f[l], v[l])
            if l == self.minlevel:
                self.residual(l, v[l], u[l], f[l])
                return np.linalg.norm(v[l])
        else:
            u[l] = self.smoothpre(l, u[l], f[l], v[l])
            v[l] = self.residual(l, v[l], u[l], f[l])
            f[l+1] = self.driver.restrict(self.grids[l], self.grids[l+1], f[l+1], v[l])
            self.driver.dirichletzero(self.grids[l+1], f[l+1])
            u[l+1].fill(0)
            for i in range(self.cycle):
                self.step(l+1, u, f, v)
            v[l].fill(0)
            self.driver.prolongate(self.grids[l], self.grids[l+1], v[l], u[l+1])
            self.driver.dirichletzero(self.grids[l], v[l])
            self.update(l,u[l], v[l])
            self.smoothpost(l, u[l], f[l], v[l])
            if l == self.minlevel: return np.linalg.norm(v[l])

#=================================================================#
class FdDriver:
    def _getlevelmax(self, n):
        import sympy
        levelmax=10000
        for i in n:
            try:
                levelmax= min(levelmax, sympy.factorint(int(i-1))[2])
            except:
                levelmax=1
        return levelmax
    def _jacobi(self, matrix, omega=0.8):
        D = 0.8/matrix.diagonal()
        return splinalg.LinearOperator(matrix.shape, lambda x: D*x)
    def _ilu(self, matrix):
        ilu = splinalg.spilu(matrix.tocsc(), fill_factor=2)
        return splinalg.LinearOperator(matrix.shape, ilu.solve)
    def __init__(self, gridf):
        from strucfem import matrix, grid
        n = gridf.n
        levelmax = self._getlevelmax(n)
        print(f"levelmax = {levelmax} nfine = {gridf.nall()}")
        self.grids, self.matrices = [], []
        self.grids.append(gridf)
        for l in range(levelmax):
            n = (n-1)//2+1
            self.grids.append(grid.Grid(n=n, bounds=gridf.bounds))
        for grid in self.grids:
            self.matrices.append(matrix.createMatrixDiff(grid))
    def getMeshesAndMatrices(self): return self.grids, self.matrices
    def newVector(self, grid): return np.zeros(grid.nall())
    def newPreconditioner(self, matrix):
        return self._ilu(matrix)
        return self._jacobi(matrix)
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
        return transfer.interpolate(gridf=gridf, gridc=gridc, unew=v, uold=u)


#=================================================================#
if __name__ == '__main__':
    import simfempy.tools.analyticalsolution as anasol
    from strucfem import matrix, grid, plotgrid
    import matplotlib.pyplot as plt

    d, l = 2, 2
    expr = ''
    for i in range(d): expr += f"log(1+x{i}**2)*"
    expr = expr[:-1]
    # expr = '3+x1+x0'
    uex = anasol.AnalyticalSolution(d, expr)
    n = np.array(d*[2**l+1])
    # n[0] = 4**l+1
    grid = grid.Grid(n=n, bounds=d*[[1,3]])
    print(f"uex = {uex} grid={grid}")
    b = matrix.createRhsVectorDiff(grid, uex)
    fd = FdDriver(grid)
    mg = MultiGrid(fd)
    u,res = mg.solve(b)
    plotgrid.plot(grid, u=u)
    plt.show()
