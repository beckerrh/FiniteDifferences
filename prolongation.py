import numpy as np
import grid
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
    assert grid.dim == 2
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
def interpolate2(grid, gridold, uold):
    nold = gridold.n
    assert grid.dim == 2
    # uold = uold.reshape(nold)
    # print(f"uold={uold.shape}")
    uold = uold.ravel()
    unew = np.zeros(grid.nall())
    nxO, nyO = nold[0], nold[1]
    nxN, nyN = grid.n[0], grid.n[1]

    iO = np.arange(uold.shape[0])
    ix = (iO//nyO).reshape(nold)
    iy = (iO%nyO).reshape(nold)
    ind1d = [np.arange(nold[i]) for i in range(gridold.dim)]
    mg = np.array(np.meshgrid(*ind1d, indexing='ij'))
    # print(f"ix=\n{ix}\n mg[0]=\n{mg[0]}\n")
    assert np.all(ix == mg[0])
    assert np.all(iy == mg[1])

    stridesO = gridold.strides()
    stridesN = grid.strides()

    # iN = nyN*2*ix + 2*iy
    # iO = nyO*ix + iy
    # iN2 = stridesN[0]*(2*mg[0]) + 2*mg[1]
    # iN2 = np.einsum('i,,i...->...', stridesN, 2 ,mg)
    # assert np.all(iN == iN2)
    # iO2 = stridesO[0]*mg[0] + mg[1]
    # iO2 = np.einsum('i,i...->...', stridesO, mg)
    # assert np.all(iO == iO2)

    iN = np.einsum('i,,i...->...', stridesN, 2 ,mg)
    iO = np.einsum('i,i...->...', stridesO, mg)
    unew[iN.ravel()] += uold[iO.ravel()]

    # for i in range(grid.dim):
    for i in range(1):
        mgi = np.take(mg, ind1d[i][1:], axis=i+1)
        iN = np.einsum('i,,i...->...', stridesN, 2 ,mgi) - stridesN[0]
        iO = np.einsum('i,i...->...', stridesO, mgi)
        unew[iN.ravel()] += 0.5*uold[iO.ravel()]
        # mgi = np.take(mg, ind1d[i][:-1], axis=i+1)
        # iN = np.einsum('i,,i...->...', stridesN, 2 ,mgi) + stridesN[0]
        # iO = np.einsum('i,i...->...', stridesO, mgi)
        # unew[iN.ravel()] += 0.5*uold[iO.ravel()]

    # iN = nyN*(2*ix[1:]-1) + 2*iy[1:]
    # iO = nyO*ix[1:] + iy[1:]
    # mgi = np.take(mg,ind1d[0][1:], axis=1)
    # # print(f"ix[1:]=\n{ix[1:]}\nmgi[0]\n{mgi[0]}\n")
    # # assert np.all(ix[1:]==mgi[0])
    # iN2 = np.einsum('i,,i...->...', stridesN, 2 ,mgi) - stridesN[0]
    # assert np.all(iN == iN2)
    # iO2 = np.einsum('i,i...->...', stridesO, mgi)
    # if not np.all(iO == iO2): print(f"iO=\n{iO}\niO2\n{iO2}\n")
    # unew[iN.ravel()] += 0.5*uold[iO.ravel()]
    #
    iN = nyN*(2*ix[:-1]+1) + 2*iy[:-1]
    iO = nyO*ix[:-1] + iy[:-1]
    unew[iN.ravel()] += 0.5*uold[iO.ravel()]
    #
    iN = nyN*2*(ix[:,1:]) + 2*iy[:,1:]-1
    iO = nyO*ix[:,1:] + iy[:,1:]
    unew[iN.ravel()] += 0.5*uold[iO.ravel()]
    #
    iN = nyN*(2*ix[:,:-1]) + 2*iy[:,:-1]+1
    iO = nyO*ix[:,:-1] + iy[:,:-1]
    unew[iN.ravel()] += 0.5*uold[iO.ravel()]


    iN = nyN*(2*ix[:-1,:-1]+1) + 2*iy[:-1,:-1]+1
    iO = nyO*ix[:-1,:-1] + iy[:-1,:-1]
    unew[iN.ravel()] += 0.25*uold[iO.ravel()]

    iN = nyN*(2*ix[1:,:-1]-1) + 2*iy[1:,:-1]+1
    iO = nyO*ix[1:,:-1] + iy[1:,:-1]
    unew[iN.ravel()] += 0.25*uold[iO.ravel()]


    iN = nyN*(2*ix[:-1,1:]+1) + 2*iy[:-1,1:]-1
    iO = nyO*ix[:-1,1:] + iy[:-1,1:]
    unew[iN.ravel()] += 0.25*uold[iO.ravel()]

    iN = nyN*(2*ix[1:,1:]-1) + 2*iy[1:,1:]-1
    iO = nyO*ix[1:,1:] + iy[1:,1:]
    unew[iN.ravel()] += 0.25*uold[iO.ravel()]

    # print(f"unew=\n{unew.reshape(grid.n)}\n")
    # print(f"iO=\n{iO} ix=\n{ix} iy=\n{iy} iN=\n{iN}")
    # unew = unew.reshape(grid.n)
    # print(f"unew.shape = {unew.shape}")
    # show(grid, unew)
    return unew.ravel()
#=================================================================#
def testprolongation(ns, bounds, expr):
    import time
    d = len(bounds)
    u = anasol.AnalyticalSolution(d, expr)
    print(f"u={u}")
    g = grid.Grid(ns[0], bounds=bounds)
    uold = u(g.coord())
    fcts = ['interpolate1','interpolate2']
    times = {}
    for fct in fcts: times[fct] = []
    ns = ns[1:]
    N = []
    for n in ns:
        gold = g
        g = grid.Grid(n=n, bounds=bounds)
        N.append(g.nall())
        args = "(grid = g, gridold = gold, uold=uold)"
        for fct in fcts:
            t0 = time.time()
            unew  = eval(fct+args)
            times[fct].append(time.time()-t0)
            assert np.linalg.norm(unew.reshape(g.n)-u(g.coord()))<1e-12
        uold = unew
    for fct in fcts:
        plt.plot(np.log(N), times[fct], 'x-', label=fct)
    plt.legend()
    plt.xlabel("log(n)")
    plt.xlabel("t")
    plt.show()


if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    # ns = [5, 9, 17, 33, 65, 129, 257, 513, 1025]
    ns = [ [3,5], [5,9], [9,17], [17,33] ]
    expr = 'x+pi*y + 7*x*y'
    testprolongation(ns, bounds=[[-1,1], [-1,1]], expr=expr)

    # nc = 3
    # nf = 2*nc-1
    # indc = np.arange(nc**2)
    # ixc = (indc//nc).reshape(nc,nc)
    # iyc = (indc%nc).reshape(nc,nc)
    # print(f"indc=\n{indc}\n ixc=\n{ixc}\n iyc=\n{iyc}")
    # indf = nf*(2*ixc) + 2*iyc
    # print(f"indf=\n{indf}\n")
    # x = np.arange(n**2).reshape((n,n))
    # print(f"x={x}")
    # y = -1*np.ones((2*n+1,2*n+1))
    # ix, iy = np.arange(n), np.arange(n)
    # ix2 = 2 * ix+1
    # iy2 = 2 * iy+1
    # y[ix2, iy2] = x[ix,iy]
    # print(f"x[ix,iy]={x[ix,iy]}")
    # print(f"y[ix2, iy2]={y[ix2, iy2]}")
    # y = y[1:-1,1:-1]
    # print(f"x=\n{x} y=\n{y}")

