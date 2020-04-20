import numpy as np
import matplotlib.pyplot as plt
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, errors, solve, transfer

#-----------------------------------------------------------------#
def test(d=1):
    n = np.random.randint(low=3,high=5, size=d)
    n = 3*np.ones(d, dtype=int)
    bounds = d*[[-1,1]]
    expr = ''
    for i in range(d): expr += f"cos(pi*x{i})+"
    expr = expr[:-1]
    uex = anasol.AnalyticalSolution(d, expr)
    solvers = ["direct", "pyamg", "mg"]
    N, errs, its, ts = [], {}, {}, {}
    for solver in solvers:
        errs[solver], its[solver], ts[solver] = [], [], []
    gold = None
    for k in range(2*(5-d)):
        n = 2 * n - 1
        g = grid.Grid(n=n, bounds=bounds)
        N.append(g.nall())
        A = matrix.createMatrixDiff(g)
        b = matrix.createRhsVectorDiff(g, uex)
        if gold == None: u0 = np.zeros_like(b)
        else: u0 = transfer.interpolate(gridf=g, gridc=gold, uold=uold)
        for solver in solvers:
            u, t, it = solve.solve(solver, A, b, x0=u0, grid=g)
            err = errors.errorl2(g, u, uex)
            errs[solver].append(err)
            its[solver].append(it)
            ts[solver].append(t)
        uold, gold = u, g
    print(f"its={its}")
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel(r'$\log_{10}(e)$')
    ax.set_title(f"{expr}")
    for solver in solvers:
        ax.plot(np.log10(N), np.log10(errs[solver]), '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(312)
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel(r'it')
    ax.set_title(f"iter")
    for solver in solvers:
        ax.plot(np.log10(N), its[solver], '-x', label=f"{solver}")
    ax.legend()
    ax = fig.add_subplot(313)
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel(r't')
    ax.set_title(f"time")
    for solver in solvers:
        ax.plot(np.log10(N), np.log10(ts[solver]), '-x', label=f"{solver}")
    ax.legend()
    plt.show()

#=================================================================#
if __name__ == '__main__':
    for d in range(1,4): test(d=d)