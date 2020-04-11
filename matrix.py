import numpy as np
import scipy.sparse as scsp
import grid
import simfempy.tools.analyticalsolution as anasol
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

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
def test(n, expr, bounds, show=False):
    g = grid.Grid(n=n, bounds=bounds)
    for i in range(g.dim):
        g.bdrycond[i][0] = g.bdrycond[i][1] = 'dirichlet'
    A = createMatrixDiff(g)
    u = anasol.AnalyticalSolution(g.dim, expr)
    b = createRhsVectorDiff(g, u)
    uh = linalg.spsolve(A,b)
    if show: plot(grid=g, uh=uh, u=u, plot_error=True)
    return errorl2(grid=g, uh=uh, u=u)

def testerror(ns, bounds, expr):
    from scipy import stats
    import time
    errs=[]
    times=[]
    d = len(bounds)
    for i,n in enumerate(ns):
        t0 = time.time()
        err = test(n=d*[n], bounds=bounds, expr=expr, show=False)
        errs.append(err)
        t1 = time.time()
        times.append(t1-t0)
    slope, ic, r, p, stderr  = stats.linregress(np.log(ns), np.log(errs))
    # print(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel(r'log(n)')
    ax.set_ylabel(r'log(e)')
    ax.set_title(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    ax.loglog(ns, errs, '-x')
    ax = fig.add_subplot(212)
    ax.set_xlabel(r'log(n)')
    ax.set_ylabel(r't')
    ax.set_title(f"nall = {ns[-1]**d}")
    # le premier est trop grand !!
    ax.plot(np.log(ns[1:]), times[1:], '-x')
    print("times", times)
    plt.show()

if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    ns = [5, 9, 17, 33, 65, 129, 267, 523, 1045]
    # testerror(ns, bounds=[[0,1]], expr='cos(pi*x)')
    # testerror(ns[:-2], bounds=[[0,1], [0,1]], expr='cos(pi*x)*cos(pi*y)')
    testerror(ns[:-3], bounds=[[0,1], [0,1], [0,1]], expr='cos(pi*x)*cos(pi*y)*cos(pi*z)')
    # err = test(n=[11], bounds=[[1,3]], expr='x*(1-x)+1', show=True)
    # print(f"errl2={err}")