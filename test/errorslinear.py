import numpy as np
import scipy.sparse.linalg as splinalg
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, errors

#-----------------------------------------------------------------#
def test(d=1):
    n = np.random.randint(low=4,high=8, size=d)
    bounds = d*[[-1,1]]
    g = grid.Grid(n=n, bounds=bounds)
    expr = ''
    for i in range(d): expr += f"{np.random.randint(low=1,high=9)}*x{i}+"
    uex = anasol.AnalyticalSolution(expr[:-1], dim=g.dim)
    A = matrix.createMatrixDiff(g)
    b = matrix.createRhsVectorDiff(g, uex)
    u = splinalg.spsolve(A, b)
    err = errors.errorl2(g, u, uex)
    print(f"u={uex} grid={g}\nerr = {err:10.4e}")

#=================================================================#
if __name__ == '__main__':
    for d in range(1,6): test(d=d)