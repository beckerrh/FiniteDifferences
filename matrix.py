import numpy as np
import scipy.sparse as scsp
import grid

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
def createVectorDiff1d(grid, f, u):
    """
    """
    x = grid.x()
    b = f(x)
    # print(f"x={x} b={b}")
    if grid.bdrycond[0][0] == 'dirichlet': b[0] = u(x[0])
    if grid.bdrycond[0][1] == 'dirichlet': b[-1] = u(x[-1])
    return b


#=================================================================#

def test1d():
    import scipy.sparse.linalg as linalg
    import matplotlib.pyplot as plt
    g = grid.Grid(n=[17], length=[[1,3]])
    g.bdrycond[0][0] = 'dirichlet'
    g.bdrycond[0][1] = 'dirichlet'
    A = createMatrixDiff1d(g)
    # print("A=", A.toarray())
    u = np.vectorize(lambda x: np.cos(np.pi*x))
    f = np.vectorize(lambda x: np.pi**2*np.cos(np.pi*x))
    # u = np.vectorize(lambda x: x*(1-x)+1)
    # f = np.vectorize(lambda x: 2)
    b = createVectorDiff1d(g, f, u)
    uh = linalg.spsolve(A,b)
    x = g.x()
    plt.plot(x,uh, '-x', x, u(x))
    plt.show()

if __name__ == '__main__':
    test1d()