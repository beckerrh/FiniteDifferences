import numpy as np
import scipy.sparse as scsp
import grid, transfer
import simfempy.tools.analyticalsolution as anasol
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import pyamg

#-----------------------------------------------------------------#
def interpolate(grid, uold):
    if uold is None:
        return np.zeros(grid.nall())
    nold = (np.array(grid.n, dtype=int)-1)//2 +1
    # print(f"uold.shape={uold.shape} n={np.prod(nold)} nold={nold}")
    # print(f"grid.n={grid.n} grid.nall={grid.nall()}")
    assert grid.dim == 2
    uold = uold.reshape(nold)
    unew = np.zeros(grid.n+2)
    # print(f"uold.shape={uold.shape} unew.shape={unew.shape}")
    for ix in range(nold[0]):
        ix2 = 2 * ix+1
        for iy in range(nold[1]):
            iy2 = 2 * iy + 1
            unew[ix2, iy2] += uold[ix, iy]
            unew[ix2 - 1, iy2] += 0.5 * uold[ix, iy]
            unew[ix2 + 1, iy2] += 0.5 * uold[ix, iy]
            unew[ix2, iy2 - 1] += 0.5 * uold[ix, iy]
            unew[ix2, iy2 + 1] += 0.5 * uold[ix, iy]
            unew[ix2 + 1, iy2 - 1] += 0.25 * uold[ix, iy]
            unew[ix2 - 1, iy2 + 1] += 0.25 * uold[ix, iy]
            unew[ix2 - 1, iy2 - 1] += 0.25 * uold[ix, iy]
            unew[ix2 + 1, iy2 + 1] += 0.25 * uold[ix, iy]
    unew = unew[1:-1,1:-1]
    # x = grid.coord()
    # cnt = plt.contour(x[0], x[1], unew)
    # plt.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
    # plt.show()
    return unew.ravel()


#-----------------------------------------------------------------#
def createMatrixDiff2d(grid):
    """
    """
    nx, ny = grid.n[0], grid.n[1]
    n = nx*ny
    dx, dy = grid.dx[0], grid.dx[1]
    diag = (2/dx/dx+2/dy/dy)*np.ones(shape=(nx,ny))
    d0P = (-1/dy/dy)*np.ones(shape=(nx,ny))
    d0M = (-1/dy/dy)*np.ones(shape=(nx,ny))
    dP0 = (-1/dx/dx)*np.ones(shape=(nx,ny))
    dM0 = (-1/dx/dx)*np.ones(shape=(nx,ny))

    # mettre a zero ce qui sont dehors
    d0M[:,0] = 0
    d0P[:,-1] = 0
    dM0[0,:] = 0
    dP0[-1,:] = 0
    # print("d0M", d0M)
    # print("d0M", d0M.reshape(nx*ny))

    if grid.bdrycond[0][0] == 'dirichlet':
        diag[0,:] = 1
        d0M[0,:] = d0P[0,:]= dM0[0,:]= dP0[0,:] = 0
    if grid.bdrycond[0][1] == 'dirichlet':
        diag[-1,:] = 1
        d0M[-1, :] = d0P[-1, :] = dM0[-1, :] = dP0[-1, :] = 0
    if grid.bdrycond[1][0] == 'dirichlet':
        diag[:,0] = 1
        d0M[:,0] = d0P[:,0] = dM0[:,0] = dP0[:,0] = 0
    if grid.bdrycond[1][1] == 'dirichlet':
        diag[:,-1] = 1
        d0M[:,-1] = d0P[:,-1] = dM0[:,-1] = dP0[:,-1] = 0
    diagonals = [diag.reshape(nx*ny)]
    offsets = [0]
    # print("d0M",d0M)
    # print("shaores memory ?", np.shares_memory(d0M,d0Mrs))
    diagonals.append(d0M.reshape(nx*ny)[1:])
    offsets.append(-1)
    diagonals.append(d0P.reshape(nx*ny)[:-1])
    offsets.append(1)
    diagonals.append(dM0.reshape(nx*ny)[nx:])
    offsets.append(-nx)
    diagonals.append(dP0.reshape(nx*ny)[:-nx])
    offsets.append(nx)
    A = scsp.diags(diagonals=diagonals, offsets=offsets)
    # print("A=\n", A.toarray())
    return A.tocsr()
#-----------------------------------------------------------------#
def createMatrixDiff(grid):
    """
    """
    n, ds, dim, nall = grid.n, grid.dx, grid.dim, grid.nall()
    sdiag = np.sum(2/ds**2)
    soff = -1/ds**2
    diag = sdiag*np.ones(shape=n)
    dP, dM = np.empty(shape=(dim,*n)), np.empty(shape=(dim,*n))
    for i in range(dim):
        dP[i] = dM[i] = soff[i]
    # mettre a zero ce qui sont dehors
    for i in range(dim):
        np.moveaxis(dP[i], i, 0)[-1] = 0
        np.moveaxis(dM[i], i, 0)[ 0] = 0
    # mettre a zero pour Dirichlet
    for i in range(dim):
        if grid.bdrycond[i][0] == 'dirichlet':
            np.moveaxis(diag, i, 0)[0] = 1
            for j in range(dim):
                np.moveaxis(dP[j], i, 0)[0] = np.moveaxis(dM[j], i, 0)[0] =0
        if grid.bdrycond[i][1] == 'dirichlet':
            np.moveaxis(diag, i, 0)[-1] = 1
            for j in range(dim):
                np.moveaxis(dP[j], i, 0)[-1] = np.moveaxis(dM[j], i, 0)[-1] =0
    diagonals = [diag.reshape(nall)]
    offsets = [0]
    for i in range(dim):
        stride = int(np.prod(list(reversed(n))[:i-1]))
        diagonals.append(dM[i].reshape(nall)[stride:])
        offsets.append(-stride)
        diagonals.append(dP[i].reshape(nall)[:-stride])
        offsets.append(stride)
    A = scsp.diags(diagonals=diagonals, offsets=offsets)
    # print("A=\n", A.toarray())
    return A.tocsr()
#-----------------------------------------------------------------#
def createRhsVectorDiff(grid, u):
    """
    """
    x = grid.coord()
    d = grid.dim
    b = np.zeros(x[0].shape)
    for i in range(d):
        b -= u.dd(i,i,x)
    for i in range(d):
        if grid.bdrycond[i][0] == 'dirichlet':
            bs, xs = np.moveaxis(b, i, 0), np.moveaxis(x, i+1, 1)
            bs[ 0] = u(xs[:, 0])
        if grid.bdrycond[i][1] == 'dirichlet':
            bs, xs = np.moveaxis(b, i, 0), np.moveaxis(x, i + 1, 1)
            bs[-1] = u(xs[:, -1])
    return b.ravel()
#-----------------------------------------------------------------#
def errorl2(grid, uh, u, compute_error=False):
    """
    """
    x = grid.coord()
    n = grid.nall()
    errl2 = np.linalg.norm(uh-u(x).ravel())/np.sqrt(n)
    if not compute_error: return errl2
    return errl2, uh-u(x)
#-----------------------------------------------------------------#
def plot(grid, uh, u=None, plot_error=False):
    """
    """
    x = grid.coord()
    d = grid.dim
    if d==1:
        plt.plot(x.ravel(), uh, '-xb')
        if plot_error:
            if u is None: raise ValueError("Problem: need exact solution")
            plt.plot(x.ravel(), u(x), '-r')
        plt.show()
    elif d==2:
        x = grid.coord()
        cnt = plt.contour(x[0], x[1], uh.reshape(grid.n))
        plt.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
        plt.show()

#=================================================================#
def test(g, expr, show=False, solver='scsp', uold=None, gold=None):
    for i in range(g.dim):
        g.bdrycond[i][0] = g.bdrycond[i][1] = 'dirichlet'
    A = createMatrixDiff(g)
    u = anasol.AnalyticalSolution(g.dim, expr)
    b = createRhsVectorDiff(g, u)
    niter = -1
    if solver == 'scsp':
        uh = linalg.spsolve(A,b)
    elif solver == 'pyamg':
        if gold == None: u0 = np.zeros_like(b)
        else: u0 = transfer.interpolate(g, gold, uold)
        res = []
        B = np.ones((A.shape[0], 1))
        SA_build_args = {
            'max_levels': 10,
            'max_coarse': 10,
            'coarse_solver': 'lu',
            'symmetry': 'hermitian'}
        # smooth = ('energy', {'krylov': 'cg'})
        # strength = [('evolution', {'k': 2, 'epsilon': 0.2})]
        presmoother = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2})
        postsmoother = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2})
        ml = pyamg.smoothed_aggregation_solver(A, B=B, **SA_build_args)
        SA_solve_args = {'cycle': 'V', 'maxiter': 200, 'tol': 1e-10}
        uh = ml.solve(b=b, x0=u0, residuals=res, **SA_solve_args)
        # uh = pyamg.solve(A, b, verb=0, tol=1e-10, x0=u0, residuals=res)
        niter = len(res)
    else:
        raise KeyError(f"Problem: unknown solver {solver}")
    if show: plot(grid=g, uh=uh, u=u, plot_error=True)
    return errorl2(grid=g, uh=uh, u=u), uh, niter

#=================================================================#
def testerror(ns, bounds, expr):
    from scipy import stats
    import time
    d = len(bounds)
    solvers = ['scsp', 'pyamg']
    # solvers = ['scsp']
    errs, times, niters = {}, {}, {}
    times={}
    for solver in solvers:
        times[solver], errs[solver], niters[solver] = [], [], []
    uold, gold = None, None
    N = []
    for i,n in enumerate(ns):
        g = grid.Grid(n=n, bounds=bounds)
        N.append(g.nall())
        for solver in solvers:
            t0 = time.time()
            err, u, niter = test(g, expr=expr, show=False, solver=solver, uold=uold, gold=gold)
            if solver == 'pyamg': uold, gold = u, g
            errs[solver].append(err)
            niters[solver].append(niter)
            t1 = time.time()
            times[solver].append(t1-t0)
    slope, ic, r, p, stderr  = stats.linregress(np.log(N), np.log(errs[solvers[0]]))
    # print(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_xlabel(r'log(N)')
    ax.set_ylabel(r'log(e)')
    ax.set_title(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    for solver in solvers:
        ax.loglog(N, errs[solver], '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(312)
    ax.set_xlabel(r'log(N)')
    ax.set_ylabel(r'niter')
    ax.set_title(f"nall = {N[-1]}")
    # le premier est trop grand !!
    print(f"niters {niters}")
    for solver in solvers:
        if np.any(np.array(niters[solver]) != -1):
            ax.plot(np.log(N), niters[solver], '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(313)
    ax.set_xlabel(r'log(N)')
    ax.set_ylabel(r't')
    ax.set_title(f"dim = {g.dim}")
    # le premier est trop grand !!
    for solver in solvers:
        ax.plot(np.log(N[1:]), times[solver][1:], '-x', label=f"{solver}")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    d = 2
    if d==2:
        ns = [np.array([3,3])]
        for k in range(10): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x)*cos(pi*y)'
    elif d==3:
        ns = [np.array([3,3,3])]
        for k in range(5): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x)*cos(pi*y)*cos(pi*z)'
    elif d==4:
        ns = [np.array([3,3,3,3])]
        for k in range(4): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x0)*cos(pi*x1)*cos(pi*x2)*cos(pi*x3)'
    testerror(ns[:-1], bounds=d*[[-1,1]], expr=expr)
