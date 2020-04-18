import numpy as np
import simfempy.tools.analyticalsolution as anasol
from strucfem import transfer, grid

#-----------------------------------------------------------------#
def errorl2(grid, u, uex, compute_error=False):
    """
    """
    x, xc, d, u = grid.coord(), grid.coordCenters(), np.sqrt(grid.volumeK()), u.ravel()
    errl2 = d*np.linalg.norm(u-uex(x).ravel())
    uc = transfer.tocell(grid,u)
    # print(f"u={u} uc={uc}")
    # print(f"x = {x} xc={xc} uexc={uex(xc)}")
    errl2 += d*np.linalg.norm(uc-uex(xc).ravel())
    if not compute_error: return errl2
    return errl2, u-uex(x)

#-----------------------------------------------------------------#
def test(d=1):
    n = np.random.randint(low=3,high=4, size=d)
    bounds = d*[[-1,1]]
    g = grid.Grid(n=n, bounds=bounds)
    expr = ''
    for i in range(d): expr += f"{np.random.randint(low=1,high=9)}*x{i}+"
    uex = anasol.AnalyticalSolution(g.dim, expr[:-1])
    u = uex(g.coord())
    err = errorl2(g, u, uex)
    print(f"u={uex} grid={g}\nerr = {err:10.4e}")
#=================================================================#
if __name__ == '__main__':
    for d in range(1,5):
        test(d=d)