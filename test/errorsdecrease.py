import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import simfempy.tools.analyticalsolution as anasol
from strucfem import matrix, grid, errors, solve, transfer

#-----------------------------------------------------------------#
def test(d=1):
    n = np.random.randint(low=3,high=4, size=d)
    bounds = d*[[-1,1]]
    expr = ''
    for i in range(d): expr += f"cos(pi*x{i})+"
    expr = expr[:-1]
    uex = anasol.AnalyticalSolution(expr,dim=d)
    N, errs, its, ts = [], [], [], []
    gold = None
    for k in range(2*(6-d)):
        n = 2 * n - 1
        g = grid.Grid(n=n, bounds=bounds)
        A = matrix.createMatrixDiff(g)
        b = matrix.createRhsVectorDiff(g, uex)
        if gold == None: u0 = np.zeros_like(b)
        else: u0 = transfer.interpolate(gridf=g, gridc=gold, uold=uold)
        u, t, it = solve.solve("pyamg",A, b, x0=u0)
        err = errors.errorl2(g, u, uex)
        uold, gold = u, g
        N.append(g.nall())
        errs.append(err)
        its.append(it)
        ts.append(t)
        # print(f"u={uex} grid={g}\nerr = {err:10.4e}")
    slope, ic, r, p, stderr  = stats.linregress(np.log10(N), np.log10(errs))
    # print(f"y = {slope:4.2f} * x  {ic:+4.2f}")
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel(r'$\log_{10}(e)$')
    ax.set_title(f"{expr}")
    ax.plot(np.log10(N), np.log10(errs), '-x', label=f"err")
    ax.plot(np.log10(N), ic+slope*np.log10(N), '-k', label=f"y = {slope:4.2f} * x  {ic:+4.2f}")
    ax.legend()
    ax = fig.add_subplot(212)
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel(r'it', color='b')
    p0, = ax.plot(np.log10(N), its, '-xb', label=f"it")
    ax.set_title(f"iter/time")
    axt = ax.twinx()
    p1, = axt.plot(np.log10(N), np.log10(ts), '-xr', label=f"t")
    axt.set_ylabel(r'$\log_{10}(t)$', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    axt.tick_params(axis='y', labelcolor='r')
    plt.legend(handles=[p0, p1])
    plt.show()

#=================================================================#
if __name__ == '__main__':
    for d in range(1,5): test(d=d)