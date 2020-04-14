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
def interpolate1(grid, uold):
    if uold is None:
        return np.zeros(grid.nall())
    nold = (np.array(grid.n, dtype=int)-1)//2 +1
    assert grid.dim == 2
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
    show(grid, unew)
    return unew.ravel()
#-----------------------------------------------------------------#
def interpolate2(grid, gridold, uold):
    nold = gridold.n
    assert grid.dim == 2
    uold = uold.reshape(nold)
    print(f"uold={uold.shape}")
    uold = uold.ravel()
    unew = np.zeros(grid.nall())
    nxO, nyO = nold[0], nold[1]
    nxN, nyN = grid.n[0], grid.n[1]

    iO = np.arange(uold.shape[0])
    ix = (iO//nyO).reshape(nold)
    iy = (iO%nyO).reshape(nold)
    iN = (nyN*2*ix + 2*iy).ravel()
    # print(f"iN=\n{iN}\n iO=\n{iO}\n")
    unew[iN] += uold[iO]
    # print(f"unew=\n{unew.reshape(grid.n)}\n")

    iN = (nyN*(2*ix[1:]-1) + 2*iy[1:]).ravel()
    iO = (nyO*(ix[1:]) + iy[1:]).ravel()
    unew[iN] += 0.5*uold[iO]

    iN = (nyN*(2*ix[:-1]+1) + 2*iy[:-1]).ravel()
    iO = (nyO*(ix[:-1]) + iy[:-1]).ravel()
    unew[iN] += 0.5*uold[iO]

    iN = (nyN*2*(ix[:,1:]) + 2*iy[:,1:]-1).ravel()
    iO = (nyO*ix[:,1:] + iy[:,1:]).ravel()
    unew[iN] += 0.5*uold[iO]

    iN = (nyN*(2*ix[:,:-1]) + 2*iy[:,:-1]+1).ravel()
    iO = (nyO*ix[:,:-1] + iy[:,:-1]).ravel()
    unew[iN] += 0.5*uold[iO]


    iN = (nyN*(2*ix[:-1,:-1]+1) + 2*iy[:-1,:-1]+1).ravel()
    iO = (nyO*ix[:-1,:-1] + iy[:-1,:-1]).ravel()
    unew[iN] += 0.25*uold[iO]

    iN = (nyN*(2*ix[1:,:-1]-1) + 2*iy[1:,:-1]+1).ravel()
    iO = (nyO*ix[1:,:-1] + iy[1:,:-1]).ravel()
    unew[iN] += 0.25*uold[iO]


    iN = (nyN*(2*ix[:-1,1:]+1) + 2*iy[:-1,1:]-1).ravel()
    iO = (nyO*ix[:-1,1:] + iy[:-1,1:]).ravel()
    unew[iN] += 0.25*uold[iO]

    iN = (nyN*(2*ix[1:,1:]-1) + 2*iy[1:,1:]-1).ravel()
    iO = (nyO*ix[1:,1:] + iy[1:,1:]).ravel()
    unew[iN] += 0.25*uold[iO]

    print(f"unew=\n{unew.reshape(grid.n)}\n")


    # print(f"iO=\n{iO} ix=\n{ix} iy=\n{iy} iN=\n{iN}")




    unew = unew.reshape(grid.n)
    print(f"unew.shape = {unew.shape}")
    show(grid, unew)
    return unew.ravel()
#=================================================================#
def testprolongation(ns, bounds, expr):
    import time
    d = len(bounds)
    u = anasol.AnalyticalSolution(d, expr)
    print(f"u={u}")
    g = grid.Grid(n=d * [ns[0]], bounds=bounds)
    uold = u(g.coord())
    times = []
    ns = ns[1:]
    for i,n in enumerate(ns):
        gold = g
        g = grid.Grid(n=d*[n], bounds=bounds)
        t0 = time.time()
        unew  = interpolate2(grid = g, gridold = gold, uold=uold)
        uold = unew
        times.append(time.time()-t0)


if __name__ == '__main__':
    # print("simfempy", simfempy.__version__)
    ns = [5, 9, 17, 33]
    ns = [3, 5]
    expr = 'x+y'
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

