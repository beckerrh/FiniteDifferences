import numpy as np
import scipy.sparse.linalg as splinalg
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, transfer, plotgrid
from linalg import mg

#=================================================================#
class MgDriver(mg.FdDriver):
    def __init__(self, grid, A, **kwargs):
        maxlevel=-10000
        if grid.dim == 1: maxlevel = -100
        super().__init__(grid, A=A, smoothers=['lgs'], maxlevel=maxlevel)

#=================================================================#
def solve(A, b, grid, x0=None, verbose=True):
    mgD = MgDriver(grid, A)
    mgS = mg.MultiGrid(mgD)
    return mgS.solve(b, x0=x0, verbose=verbose)

#=================================================================#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d, l = 3, 6
    expr = ''
    for i in range(d): expr += f"log(1+x{i}**2)*"
    expr = expr[:-1]
    uex = anasol.AnalyticalSolution(d, expr)
    bounds=d*[[-1,1]]
    grid = grid.Grid(n=np.array(d*[2**l+1]), bounds=bounds)
    print(f"uex = {uex} N={grid.nall()}")
    b = matrix.createRhsVectorDiff(grid, uex)
    A = matrix.createMatrixDiff(grid)
    u, res = solve(A, b, grid)
    plotgrid.plot(grid, u=u)
    plt.show()
