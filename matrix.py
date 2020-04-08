import numpy as np
import scipy.sparse as scsp
import grid
import simfempy.tools.analyticalsolution as anasol

#-----------------------------------------------------------------#
def createMatrixDiff1d(grid):
    """
    """
    nx = grid.n[0]
    dx = grid.dx[0]
    offdiag = -np.ones(nx - 1)/dx/dx
    left = offdiag.copy()
    right = offdiag.copy()
    diag = 2*np.ones(nx)/dx/dx
    diag[0] /= 2
    diag[-1] /= 2
    if grid.bdrycond[0][0] == 'dirichlet':
        diag[0] = 1
        right[0] = 0
    if grid.bdrycond[0][1] == 'dirichlet':
        diag[-1] = 1
        left[-1] = 0
    A = scsp.diags(diagonals=(left, diag, right), offsets=[-1,0,1])
    return A.tocsr()
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
def createVectorDiff1d(grid, u):
    """
    """
    x, = grid.coord()
    b = -u.xx(x)
    # print(f"x={x} b={b}")
    if grid.bdrycond[0][0] == 'dirichlet': b[0] = u(x[0])
    if grid.bdrycond[0][1] == 'dirichlet': b[-1] = u(x[-1])
    return b

#-----------------------------------------------------------------#
def createVectorDiff2d(grid, u):
    """
    """
    x,y = grid.coord()
    b = -u.xx(x,y) - u.yy(x,y)
    print("b", b.shape)
    # print(f"x={x} b={b}")
    if grid.bdrycond[0][0] == 'dirichlet': b[0,:] = u(x[0,:],y[0,:])
    if grid.bdrycond[0][1] == 'dirichlet': b[-1,:] = u(x[-1,:],y[-1,:])
    if grid.bdrycond[1][0] == 'dirichlet': b[:,0] = u(x[:,0],y[:,0])
    if grid.bdrycond[1][1] == 'dirichlet': b[:,-1] = u(x[:,-1],y[:,-1])
    return b.ravel()


#=================================================================#

def test1d():
    import scipy.sparse.linalg as linalg
    import matplotlib.pyplot as plt
    g = grid.Grid(n=[17], length=[[1,3]])
    g.bdrycond[0][0] = 'dirichlet'
    g.bdrycond[0][1] = 'dirichlet'
    A = createMatrixDiff1d(g)
    # print("A=", A.toarray())
    u = anasol.AnalyticalSolution('cos(pi*x)')
    u = anasol.AnalyticalSolution('x*(1-x)+1')
    b = createVectorDiff1d(g, u)
    print("b=",b)
    uh = linalg.spsolve(A,b)
    x, = g.coord()
    plt.plot(x,uh, '-x', x, u(x))
    plt.show()


def test2d():
    import scipy.sparse.linalg as linalg
    import matplotlib.pyplot as plt
    g = grid.Grid(n=[15,15], length=[[0,1],[0,1]])
    g.bdrycond[0][0] = 'dirichlet'
    g.bdrycond[0][1] = 'dirichlet'
    g.bdrycond[1][0] = 'dirichlet'
    g.bdrycond[1][1] = 'dirichlet'
    A = createMatrixDiff2d(g)
    # print("A=", A.toarray())
    # u = np.vectorize(lambda x,y: np.cos(np.pi*x)*np.cos(np.pi*y))
    # f = np.vectorize(lambda x,y: 2*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y))
    # u = np.vectorize(lambda x,y: x*(1-x)+y*(1-y)+1)
    # f = np.vectorize(lambda x,y: 4)
    # u = anasol.AnalyticalSolution('cos(pi*x)*cos(pi*y)')
    u = anasol.AnalyticalSolution('x*(1-x)+y*(1-y)+1')
    b = createVectorDiff2d(g, u)
    uh = linalg.spsolve(A,b)
    x,y = g.coord()
    # plt.plot(x,uh, '-x', x, u(x))
    cnt = plt.contour(x, y, uh.reshape(g.n[0],g.n[1]))
    plt.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
    plt.show()

if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    # test1d()
    test2d()