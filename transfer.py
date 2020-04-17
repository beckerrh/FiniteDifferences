import numpy as np
import grid, tools
import simfempy.tools.analyticalsolution as anasol
import matplotlib.pyplot as plt

#-----------------------------------------------------------------#
def show(grid, u):
    x = grid.coord()
    cnt = plt.contour(x[0], x[1], u)
    plt.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
    plt.show()
#-----------------------------------------------------------------#
def interpolate1(grid, gridold, uold):
    nold = gridold.n
    uold = uold.reshape(nold)
    unew = np.zeros(grid.n+2)
    for ix in range(nold[0]):
        ix2 = 2 * ix+1
        for iy in range(nold[1]):
            iy2 = 2 * iy + 1
            unew[ix2, iy2] += uold[ix, iy]
            unew[ix2 - 1, iy2] += 0.5 * uold[ix, iy]
            unew[ix2 + 1, iy2] += 0.5 * uold[ix, iy]
            unew[ix2, iy2 - 1] += 0.5 * uold[ix, iy]
            unew[ix2, iy2 + 1] += 0.5 * uold[ix, iy]
            unew[ix2 + 1, iy2 - 1] += 0.25 * uold[ix, iy]
            unew[ix2 - 1, iy2 + 1] += 0.25 * uold[ix, iy]
            unew[ix2 - 1, iy2 - 1] += 0.25 * uold[ix, iy]
            unew[ix2 + 1, iy2 + 1] += 0.25 * uold[ix, iy]
    unew = unew[1:-1,1:-1]
    # show(grid, unew)
    return unew.ravel()
#-----------------------------------------------------------------#
def interpolate(gridf, gridc, uold, transpose=False):
    nold = gridc.n
    uold = uold.ravel()
    if transpose:
        unew = np.zeros(gridc.nall())
        if uold.shape[0] != gridf.nall():
            raise ValueError(f"Problem interpolate(transpose={transpose}) {uold.shape[0]} != {gridf.nall()}")
    else:
        unew = np.zeros(gridf.nall())
        if uold.shape[0] != gridc.nall():
            raise ValueError(f"Problem interpolate(transpose={transpose}) u.N = {uold.shape[0]} != {gridc.nall()} = grid.N")
    ind1d = [np.arange(nold[i]) for i in range(gridc.dim)]
    mg = np.array(np.meshgrid(*ind1d, indexing='ij'))
    stridesO = gridc.strides()
    stridesN = gridf.strides()
    # print(f"stridesO={stridesO} stridesN={stridesN}")
    iN = 2*np.einsum('i,i...->...', stridesN, mg)
    iO = np.einsum('i,i...->...', stridesO, mg)
    if transpose: unew[iO.ravel()] += uold[iN.ravel()]
    else: unew[iN.ravel()] += uold[iO.ravel()]
    for k in range(1,gridf.dim+1):
        inds, sts = tools.indsAndShifts(gridf.dim, k=k)
        # print(f"k={k} inds={inds} sts={sts}")
        for ind in inds:
            for st in sts:
                mg2 = mg.copy()
                for l in range(k):
                    if st[l]==-1: mg2 = np.take(mg2, ind1d[ind[l]][1:], axis=ind[l]+1)
                    else:         mg2 = np.take(mg2, ind1d[ind[l]][:-1], axis=ind[l]+1)
                iO = np.einsum('i,i...->...', stridesO, mg2)
                iN = 2*np.einsum('i,i...->...', stridesN ,mg2)+stridesN[ind].dot(st)
                if transpose: unew[iO.ravel()] += 0.5**k*uold[iN.ravel()]
                else: unew[iN.ravel()] += 0.5 ** k * uold[iO.ravel()]
    return unew

#=================================================================#
def testprolongation(ns, bounds, expr):
    import time
    d = len(bounds)
    u = anasol.AnalyticalSolution(d, expr)
    print(f"u={u}")
    g = grid.Grid(ns[0], bounds=bounds)
    uold = u(g.coord())
    fcts = ['interpolate']
    times = {}
    for fct in fcts: times[fct] = []
    ns = ns[1:]
    N = []
    for n in ns:
        gold = g
        g = grid.Grid(n=n, bounds=bounds)
        N.append(g.nall())
        args = "(gridf = g, gridc = gold, uold=uold)"
        for fct in fcts:
            t0 = time.time()
            unew  = eval(fct+args)
            times[fct].append(time.time()-t0)
            err = np.linalg.norm(unew.reshape(g.n)-u(g.coord()))/np.sqrt(g.nall())
            if err >1e-15:
                print(f"err={err:12.4e}\n u=\n{unew}")
                print(f"uold={uold.sum()}\n u=\n{unew.sum()}")
        uold = unew
    for fct in fcts:
        plt.plot(np.log(N), np.log(times[fct]), 'x-', label=fct)
    plt.legend()
    plt.xlabel("log(n)")
    plt.ylabel("log(t)")
    plt.show()


if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    # ns = [5, 9, 17, 33, 65, 129, 257, 513, 1025]

    test2d, test3d, test4d = False, False, True
    if test2d:
        ns = [np.array([5,3])]
        for k in range(1): ns.append(2*ns[k]-1)
        expr = 'x+pi*y + 7*x*y'
        testprolongation(ns, bounds=2*[[-1,1]], expr=expr)

    if test3d:
        ns = [np.array([3,3,5])]
        for k in range(6): ns.append(2*ns[k]-1)
        expr = 'x+2*y+3*z-x*y-pi*x*z + pi**2*y*z'
        testprolongation(ns, bounds=3*[[-1,1]], expr=expr)

    if test4d:
        ns = [np.array([3,3,5,3])]
        for k in range(4): ns.append(2*ns[k]-1)
        expr = 'x0+2*x1+3*x2+4*x3-x0*x2-pi*x3*x1 + pi**2*x0*x3'
        testprolongation(ns, bounds=4*[[-1,1]], expr=expr)
