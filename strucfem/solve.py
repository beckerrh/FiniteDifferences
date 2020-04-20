import time
import numpy as np
import scipy.sparse.linalg as splinalg
import pyamg
import simfempy.tools.analyticalsolution as anasol
from strucfem import grid, matrix, errors, multigrid

#-----------------------------------------------------------------#
def solve(name, A, b, x0=None, grid=None):
    t0 = time.time()
    if name == "direct":
        x, iter = splinalg.spsolve(A, b), -1
    elif name == "mg":
        assert grid is not None
        x, res = multigrid.solve(A, b, grid=grid, x0=x0, verbose=False)
        iter = len(res)
    elif name == 'pyamg':
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
        x, iter = ml.solve(b=b, x0=x0, residuals=res, **SA_solve_args), len(res)
        # uh = pyamg.solve(A, b, verb=0, tol=1e-10, x0=u0, residuals=res)
    else:
        raise ValueError(f"Problem: unknown solver name={name}")
    t1 = time.time()
    return x, t1-t0, iter

#-----------------------------------------------------------------#
def test(solvers, d=1):
    n = np.random.randint(low=10,high=20, size=d)
    bounds = d*[[-1,1]]
    g = grid.Grid(n=n, bounds=bounds)
    expr = ''
    for i in range(d): expr += f"{np.random.randint(low=1,high=9)}*x{i}+"
    uex = anasol.AnalyticalSolution(g.dim, expr[:-1])
    A = matrix.createMatrixDiff(g)
    b = matrix.createRhsVectorDiff(g, uex)
    for solver in solvers:
        u, t, iter = solve(solver, A, b, grid=g)
        err = errors.errorl2(g, u, uex)
        print(f"grid={g}\nsolver = {solver}\terr = {err:10.4e}\t time = {t}\t iter = {iter}")
#=================================================================#
if __name__ == '__main__':
    solvers = ["direct", "pyamg", "mg"]
    solvers = ["direct", "mg"]
    test(solvers, d=2)