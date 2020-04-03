import numpy as np
import scipy.sparse as scsp

#-----------------------------------------------------------------#
def createMatrixDiff1d(nx, dirichlet='both'):
    """
    nx is the number of nodes
    """
    dx = 1/(nx-1)
    offdiag = -np.ones(nx - 1)/dx/dx
    left = offdiag
    right = offdiag
    diag = 2*np.ones(nx)/dx/dx
    diag[0] /= 2
    diag[-1] /= 2
    if dirichlet in ['left', 'both']:
        diag[0] = 1
        right[0] = 0
    if dirichlet in ['right', 'both']:
        diag[-1] = 1
        left[-1] = 0
    A = scsp.diags(diagonals=(left, diag, right), offsets=[-1,0,1])
    return A.tocsr()


#=================================================================#
if __name__ == '__main__':
    import scipy.sparse.linalg
    import matplotlib.pyplot as plt
    nx = 5
    dirichlet = 'right'
    A = createMatrixDiff1d(nx, dirichlet=dirichlet)
    print("A=",A.toarray())
    b = np.ones(nx)
    if dirichlet in ['left', 'both']: b[0] = 0
    if dirichlet in ['right', 'both']: b[-1] = 0
    x = scipy.sparse.linalg.spsolve(A,b)
    print("x=",x)
    plt.plot(x)
    plt.show()
