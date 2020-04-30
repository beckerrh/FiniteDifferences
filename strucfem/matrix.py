import numpy as np
import scipy.sparse as scsp
from strucfem import grid, transfer, errors, solve
import simfempy.tools.analyticalsolution as anasol
import matplotlib.pyplot as plt

#-----------------------------------------------------------------#
def createMatrix3pd(grid, diags):
    """
    3**d stencil
    """
    n, dim, nall = grid.n, grid.dim, grid.nall()
    diagonals = []
    offsets = []
    strides = grid.strides()
    if len(diags.shape) != dim+1: raise ValueError(f"wrong sizes dim={dim} diags.shape={diags.shape}")
    if dim==2:
        if np.any(np.array(diags.shape[:-1]) != 3): raise ValueError(f"wrong sizes dim={dim} diags.shape[:-1]={diags.shape[:-1]}")
        for ii in range(3):
            sti = [-strides[0], 0, strides[0]]
            for jj in range(3):
                stj = [-strides[1], 0, strides[1]]
                stride = sti[ii] + stj[jj]
                if stride <=0: diagonals.append(diags[ii,jj,-stride:])
                else: diagonals.append(diags[ii,jj,:stride])
                offsets.append(stride)
    print(f"offsets={offsets} strides={strides}")
    A = scsp.diags(diagonals=diagonals, offsets=offsets, shape=(nall,diags.shape[-1]))
    return A

#-----------------------------------------------------------------#
def createMatrix1p2d(grid, diag, dP, dM):
    """
    1+2*d stencil
    """
    n, dim = grid.n, grid.dim
    if len(dP) != dim or len(dM) != dim: raise ValueError(f"wrong sizes dim={dim} {len(dP)} {len(dM)}")
    diagonals = [diag.ravel()]
    offsets = [0]
    strides = grid.strides()
    for i in range(dim):
        stride = strides[i]
        diagonals.append(dM[i].ravel()[stride:])
        offsets.append(-stride)
        diagonals.append(dP[i].ravel()[:-stride])
        offsets.append(stride)
    A = scsp.diags(diagonals=diagonals, offsets=offsets)
    # print("A=\n", A.toarray())
    return A

#-----------------------------------------------------------------#
# def interpolate(grid, uold):
#     if uold is None:
#         return np.zeros(grid.nall())
#     nold = (np.array(grid.n, dtype=int)-1)//2 +1
#     # print(f"uold.shape={uold.shape} n={np.prod(nold)} nold={nold}")
#     # print(f"grid.n={grid.n} grid.nall={grid.nall()}")
#     assert grid.dim == 2
#     uold = uold.reshape(nold)
#     unew = np.zeros(grid.n+2)
#     # print(f"uold.shape={uold.shape} unew.shape={unew.shape}")
#     for ix in range(nold[0]):
#         ix2 = 2 * ix+1
#         for iy in range(nold[1]):
#             iy2 = 2 * iy + 1
#             unew[ix2, iy2] += uold[ix, iy]
#             unew[ix2 - 1, iy2] += 0.5 * uold[ix, iy]
#             unew[ix2 + 1, iy2] += 0.5 * uold[ix, iy]
#             unew[ix2, iy2 - 1] += 0.5 * uold[ix, iy]
#             unew[ix2, iy2 + 1] += 0.5 * uold[ix, iy]
#             unew[ix2 + 1, iy2 - 1] += 0.25 * uold[ix, iy]
#             unew[ix2 - 1, iy2 + 1] += 0.25 * uold[ix, iy]
#             unew[ix2 - 1, iy2 - 1] += 0.25 * uold[ix, iy]
#             unew[ix2 + 1, iy2 + 1] += 0.25 * uold[ix, iy]
#     unew = unew[1:-1,1:-1]
#     # x = grid.coord()
#     # cnt = plt.contour(x[0], x[1], unew)
#     # plt.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
#     # plt.show()
#     return unew.ravel()
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
    return A.tocsc()
#-----------------------------------------------------------------#
def createMatrixDiff(grid):
    """
    1+2*d stencil
    """
    n, ds, dim, nall, vol = grid.n, grid.dx, grid.dim, grid.nall(), grid.volumeK()
    sdiag = vol*np.sum(2/ds**2)
    soff = -vol/ds**2
    diag = sdiag*np.ones(shape=n)
    dP, dM = np.empty(shape=(dim,*n)), np.empty(shape=(dim,*n))
    for i in range(dim):
        dP[i] = dM[i] = soff[i]
    # mettre a zero ce qui sont dehors
    # for i in range(dim):
    #     np.moveaxis(dP[i], i, 0)[-1] = 0
    #     np.moveaxis(dM[i], i, 0)[ 0] = 0
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
    return createMatrix1p2d(grid, diag, dP, dM)
#-----------------------------------------------------------------#
def dirichlet(grid, v, u):
    x = grid.coord()
    d = grid.dim
    v = v.reshape(grid.n)
    for i in range(d):
        if grid.bdrycond[i][0] == 'dirichlet':
            vs, xs = np.moveaxis(v, i, 0), np.moveaxis(x, i+1, 1)
            vs[ 0] = u(xs[:, 0])
        if grid.bdrycond[i][1] == 'dirichlet':
            vs, xs = np.moveaxis(v, i, 0), np.moveaxis(x, i+1, 1)
            vs[-1] = u(xs[:, -1])
    return v.ravel()
#-----------------------------------------------------------------#
def createRhsVectorDiff(grid, u):
    """
    """
    x, d, v = grid.coord(), grid.dim, grid.volumeK()
    b = np.zeros(x[0].shape)
    for i in range(d):
        b -= u.dd(i,i,x)*v
    return dirichlet(grid, b, u)
#=================================================================#
def test(g, expr, solver='direct', uold=None, gold=None):
    for i in range(g.dim):
        g.bdrycond[i][0] = g.bdrycond[i][1] = 'dirichlet'
    A = createMatrixDiff(g)
    uex = anasol.AnalyticalSolution(g.dim, expr)
    b = createRhsVectorDiff(g, uex)
    if gold == None:
        u0 = np.zeros_like(b)
    else:
        u0 = transfer.interpolate(gridf=g, gridc=gold, uold=uold)
    u, t, iter = solve.solve(solver, A, b, x0=u0, grid=g)
    return errors.errorl2(grid=g, u=u, uex=uex), u, t, iter

#=================================================================#
def testerror(ns, bounds, expr):
    from scipy import stats
    solvers = ['mg', 'pyamg']
    # solvers = ['scsp']
    errs, times, niters = {}, {}, {}
    times={}
    for solver in solvers:
        times[solver], errs[solver], niters[solver] = [], [], []
    uold, gold = None, None
    N = []
    for i,n in enumerate(ns):
        if i: uold, gold = u, g
        g = grid.Grid(n=n, bounds=bounds)
        N.append(g.nall())
        for solver in solvers:
            err, u, t, niter = test(g, expr=expr, solver=solver, uold=uold, gold=gold)
            # if solver == 'pyamg': uold, gold = u, g
            errs[solver].append(err)
            niters[solver].append(niter)
            times[solver].append(t)
    slope, ic, r, p, stderr  = stats.linregress(np.log(N), np.log(errs[solvers[0]]))
    # print(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_xlabel(r'log10(N)')
    ax.set_ylabel(r'log(e)')
    ax.set_title(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    for solver in solvers:
        ax.plot(np.log10(N), np.log10(errs[solver]), '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(312)
    ax.set_xlabel(r'log10(N)')
    ax.set_ylabel(r'niter')
    ax.set_title(f"nall = {N[-1]}")
    # le premier est trop grand !!
    print(f"niters {niters}")
    for solver in solvers:
        if np.any(np.array(niters[solver]) != -1):
            ax.plot(np.log10(N), niters[solver], '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(313)
    ax.set_xlabel(r'log10(N)')
    ax.set_ylabel(r't')
    ax.set_title(f"dim = {g.dim}")
    # le premier est trop grand !!
    for solver in solvers:
        ax.plot(np.log10(N[1:]), times[solver][1:], '-x', label=f"{solver}")
    ax.legend()
    plt.show()

#=================================================================#
if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    d = 3
    if d==1:
        ns = [np.array([3])]
        for k in range(10): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x)'
    elif d==2:
        ns = [np.array([6,3])]
        for k in range(8): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x)*cos(pi*y)'
    elif d==3:
        ns = [np.array([3,3,4])]
        for k in range(4): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x)*cos(pi*y)*cos(pi*z)'
    elif d==4:
        ns = [np.array([3,3,3,3])]
        for k in range(4): ns.append(2*ns[k]-1)
        expr = 'cos(pi*x0)*cos(pi*x1)*cos(pi*x2)*cos(pi*x3)'
    testerror(ns[:-1], bounds=d*[[-1,1]], expr=expr)
