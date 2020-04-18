import numpy as np
from strucfem import matrix


#-----------------------------------------------------------------#
def initialcondition(nx, a):
    u0 = np.empty(nx)
    nxh = int(nx/2)
    if nx%2==1: nx1 = nxh+1
    else: nx1 = nxh
    u0[:nx1] = np.linspace(0, a, nx1)
    u0[nxh:] = np.linspace(a, 0, nx-nxh)
    return u0


#=================================================================#
if __name__ == '__main__':
    nx = 6
    a =1
    A = matrix.createMatrixDiff1d(nx)
    u0 = initialcondition(nx, a)
    # plt.plot(u0)
    # plt.show()
