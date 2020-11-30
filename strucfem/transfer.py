import numpy as np
import scipy as sp
from strucfem import grid, tools, plotgrid, matrix
import simfempy.tools.analyticalsolution as anasol
import matplotlib.pyplot as plt
# from numba import jit


#-----------------------------------------------------------------#
def tocell(grid, u):
    uc = np.zeros(grid.n-1).ravel()
    u = u.ravel()
    # print(f"uc={uc}\nu={u}")
    ind1d = [np.arange(grid.n[i]) for i in range(grid.dim)]
    mg = np.array(np.meshgrid(*ind1d, indexing='ij'))
    strides = grid.strides()
    k = grid.dim
    inds, sts = tools.indsAndShifts(grid.dim, k=k)
    # print(f"k={k} inds={inds} sts={sts}\nmg={mg}\nuc={uc}\nu={u}")
    for ind in inds:
        for st in sts:
            mg2 = mg.copy()
            for l in range(k):
                if st[l] == -1:
                    mg2 = np.take(mg2, ind1d[ind[l]][1:], axis=ind[l] + 1)
                else:
                    mg2 = np.take(mg2, ind1d[ind[l]][:-1], axis=ind[l] + 1)
            iN = np.einsum('i,i...->...', strides, mg2) + strides[ind].dot(st)
            # print(f"iN={iN}")
            uc += 0.5 ** k * u[iN.ravel()]
    # print(f"uc={uc}\nu={u}")
    return uc

#-----------------------------------------------------------------#
def adress(a):
    return a.__array_interface__['data'][0]
#-----------------------------------------------------------------#
class InterpolateSimple:
    def __init__(self, grids):
        self.help = []
        self.dim = grids[0].dim
        for grid in grids: self.help.append(np.zeros(grid.n+2))
    def interpolate(self,gridf, gridc, uold, transpose=False, unew=None, level=None):
        assert level < len(self.help)
        # print(f"level = {level} {len(self.help)}")
        dim, help = self.dim, self.help[level]
        if not np.all(help.shape == gridf.n + 2):
            raise ValueError(f"level = {level} help.shape={help.shape} != {gridf.n + 2} {gridc.n + 2}")
        if transpose:
            # print(f"transpose uold = {uold} ({id(uold)}) h=({id(help)}) {adress(help)} ")
            if unew is None: unew = np.zeros(gridc.n)
            else:
                unew.fill(0.0)
                unew = unew.reshape(gridc.n)
            help.fill(0.0)
            uold.shape = gridf.n
            if   dim == 1: help[1:-1] = uold
            elif dim == 2: help[1:-1, 1:-1] = uold
            elif dim == 3: help[1:-1, 1:-1, 1:-1] = uold
            elif dim == 4: help[1:-1, 1:-1, 1:-1, 1:-1] = uold
            _restrictsimple(dim, unew, help)
            # print(f"transpose unew = {unew} {id(unew)} h=({id(help)}) {adress(help)}")
            # help[:] = 7.0
            uold.shape = gridf.nall()
            return unew.ravel()
        else:
            # print(f"uold = {uold} {id(uold)} h=({id(help)}")
            if unew is None: unew = np.zeros(gridf.n)
            else:
                unew.fill(0.0)
                unew = unew.reshape(gridf.n)
            help.fill(0.0)
            uold.shape = gridc.n
            _interpolatesimple(dim, help, uold)
            if   dim == 1: unew[:] = help[1:-1]
            elif dim == 2: unew[:,:] = help[1:-1, 1:-1]
            elif dim == 3: unew[:,:,:] = help[1:-1, 1:-1, 1:-1]
            elif dim == 4: unew[:,:,:,:] = help[1:-1, 1:-1, 1:-1, 1:-1]
            # print(f"unew = {unew} {id(unew)} h=({id(help)}")
            # print(f"??? {uold.shape}")
            uold.shape = gridc.nall()
            # print(f"??? {uold.shape}")
            return unew.ravel()
def interpolatesimplenumba(gridf, gridc, uold, transpose=False, unew=None, level=None):
    return interpolatesimple(gridf, gridc, uold, transpose=transpose, unew=unew, level=level, numba=True)
def interpolatesimple(gridf, gridc, uold, transpose=False, unew=None, level=None, numba=False):
    nc, dim = gridc.n, gridc.dim
    if transpose:
        unew = np.zeros(gridc.n)
        uold.shape = gridf.n
        help = np.pad(uold, (1, 1), 'constant')
        if numba:
            _restrictsimplenumba(dim, unew, help)
        else:
            _restrictsimple(dim, unew, help)
        uold.shape = gridf.nall()
        return unew.ravel()
    else:
        uold = uold.reshape(gridc.n)
        unew = np.zeros(gridf.n+2)
        if numba:
            _interpolatesimplenumba(dim, unew, uold)
        else:
            _interpolatesimple(dim, unew, uold)
        if dim==1: unew = unew[1:-1]
        elif dim==2: unew = unew[1:-1, 1:-1]
        elif dim==3: unew = unew[1:-1, 1:-1, 1:-1]
        elif dim==4: unew = unew[1:-1, 1:-1, 1:-1, 1:-1]
    return unew.ravel()
    # show(grid, uf)
#-----------------------------------------------------------------#
def _interpolatesimple(dim, uf, uc):
    if dim==1:
        uf[1::2] += uc
        uf[:-1:2] += 0.5*uc
        uf[2::2] += 0.5*uc
    elif dim==2:
        uf[1::2, 1::2] += uc
        uf[:-1:2, 1::2] += 0.5*uc
        uf[2::2, 1::2] += 0.5*uc
        uf[1::2, :-1:2] += 0.5*uc
        uf[1::2, 2::2] += 0.5*uc
        uf[:-1:2, :-1:2] += 0.25*uc
        uf[2::2, :-1:2] += 0.25*uc
        uf[:-1:2, 2::2] += 0.25*uc
        uf[2::2, 2::2] += 0.25*uc
    elif dim==3:
        uf[1::2, 1::2, 1::2] += uc
        uf[:-1:2,  1::2,  1::2] += 0.5*uc
        uf[ 2::2,  1::2,  1::2] += 0.5*uc
        uf[ 1::2, :-1:2,  1::2] += 0.5*uc
        uf[ 1::2,  2::2,  1::2] += 0.5*uc
        uf[ 1::2,  1::2, :-1:2] += 0.5*uc
        uf[ 1::2,  1::2,  2::2] += 0.5*uc

        uf[:-1:2, :-1:2,  1::2] += 0.25*uc
        uf[ 2::2, :-1:2,  1::2] += 0.25*uc
        uf[:-1:2,  2::2,  1::2] += 0.25*uc
        uf[ 2::2,  2::2,  1::2] += 0.25*uc
        uf[:-1:2,  1::2, :-1:2] += 0.25*uc
        uf[ 2::2,  1::2, :-1:2] += 0.25*uc
        uf[:-1:2,  1::2,  2::2] += 0.25*uc
        uf[ 2::2,  1::2,  2::2] += 0.25*uc
        uf[ 1::2, :-1:2, :-1:2] += 0.25*uc
        uf[ 1::2,  2::2, :-1:2] += 0.25*uc
        uf[ 1::2, :-1:2,  2::2] += 0.25*uc
        uf[ 1::2,  2::2,  2::2] += 0.25*uc

        uf[ 2::2, :-1:2, :-1:2] += 0.125*uc
        uf[:-1:2, :-1:2, :-1:2] += 0.125*uc
        uf[ 2::2,  2::2, :-1:2] += 0.125*uc
        uf[:-1:2,  2::2, :-1:2] += 0.125*uc
        uf[ 2::2, :-1:2,  2::2] += 0.125*uc
        uf[:-1:2, :-1:2,  2::2] += 0.125*uc
        uf[ 2::2,  2::2,  2::2] += 0.125*uc
        uf[:-1:2,  2::2,  2::2] += 0.125*uc
    elif dim==4:
        uf[1::2, 1::2, 1::2, 1::2] += uc
        uf[:-1:2,  1::2,  1::2,  1::2] += 0.5*uc
        uf[ 2::2,  1::2,  1::2,  1::2] += 0.5*uc
        uf[ 1::2, :-1:2,  1::2,  1::2] += 0.5*uc
        uf[ 1::2,  2::2,  1::2,  1::2] += 0.5*uc
        uf[ 1::2,  1::2, :-1:2,  1::2] += 0.5*uc
        uf[ 1::2,  1::2,  2::2,  1::2] += 0.5*uc
        uf[ 1::2,  1::2,  1::2,  2::2] += 0.5*uc
        uf[ 1::2,  1::2,  1::2, :-1:2] += 0.5*uc

        uf[:-1:2, :-1:2,  1::2,  1::2] += 0.25*uc
        uf[ 2::2, :-1:2,  1::2,  1::2] += 0.25*uc
        uf[:-1:2,  2::2,  1::2,  1::2] += 0.25*uc
        uf[ 2::2,  2::2,  1::2,  1::2] += 0.25*uc
        uf[:-1:2,  1::2, :-1:2,  1::2] += 0.25*uc
        uf[ 2::2,  1::2, :-1:2,  1::2] += 0.25*uc
        uf[:-1:2,  1::2,  2::2,  1::2] += 0.25*uc
        uf[ 2::2,  1::2,  2::2,  1::2] += 0.25*uc
        uf[ 1::2, :-1:2, :-1:2,  1::2] += 0.25*uc
        uf[ 1::2,  2::2, :-1:2,  1::2] += 0.25*uc
        uf[ 1::2, :-1:2,  2::2,  1::2] += 0.25*uc
        uf[ 1::2,  2::2,  2::2,  1::2] += 0.25*uc
        uf[:-1:2,  1::2,  1::2,  2::2] += 0.25*uc
        uf[ 2::2,  1::2,  1::2,  2::2] += 0.25*uc
        uf[ 1::2, :-1:2,  1::2,  2::2] += 0.25*uc
        uf[ 1::2,  2::2,  1::2,  2::2] += 0.25*uc
        uf[ 1::2,  1::2, :-1:2,  2::2] += 0.25*uc
        uf[ 1::2,  1::2,  2::2,  2::2] += 0.25*uc
        uf[:-1:2,  1::2,  1::2, :-1:2] += 0.25*uc
        uf[ 2::2,  1::2,  1::2, :-1:2] += 0.25*uc
        uf[ 1::2, :-1:2,  1::2, :-1:2] += 0.25*uc
        uf[ 1::2,  2::2,  1::2, :-1:2] += 0.25*uc
        uf[ 1::2,  1::2, :-1:2, :-1:2] += 0.25*uc
        uf[ 1::2,  1::2,  2::2, :-1:2] += 0.25*uc

        uf[ 2::2, :-1:2, :-1:2,  1::2] += 0.125*uc
        uf[:-1:2, :-1:2, :-1:2,  1::2] += 0.125*uc
        uf[ 2::2,  2::2, :-1:2,  1::2] += 0.125*uc
        uf[:-1:2,  2::2, :-1:2,  1::2] += 0.125*uc
        uf[ 2::2, :-1:2,  2::2,  1::2] += 0.125*uc
        uf[:-1:2, :-1:2,  2::2,  1::2] += 0.125*uc
        uf[ 2::2,  2::2,  2::2,  1::2] += 0.125*uc
        uf[:-1:2,  2::2,  2::2,  1::2] += 0.125*uc
        uf[ 2::2, :-1:2,  1::2, :-1:2] += 0.125*uc
        uf[:-1:2, :-1:2,  1::2, :-1:2] += 0.125*uc
        uf[ 2::2,  2::2,  1::2, :-1:2] += 0.125*uc
        uf[:-1:2,  2::2,  1::2, :-1:2] += 0.125*uc
        uf[ 2::2, :-1:2,  1::2,  2::2] += 0.125*uc
        uf[:-1:2, :-1:2,  1::2,  2::2] += 0.125*uc
        uf[ 2::2,  2::2,  1::2,  2::2] += 0.125*uc
        uf[:-1:2,  2::2,  1::2,  2::2] += 0.125*uc
        uf[ 2::2,  1::2, :-1:2, :-1:2] += 0.125*uc
        uf[:-1:2,  1::2, :-1:2, :-1:2] += 0.125*uc
        uf[ 2::2,  1::2,  2::2, :-1:2] += 0.125*uc
        uf[:-1:2,  1::2,  2::2, :-1:2] += 0.125*uc
        uf[ 2::2,  1::2, :-1:2,  2::2] += 0.125*uc
        uf[:-1:2,  1::2, :-1:2,  2::2] += 0.125*uc
        uf[ 2::2,  1::2,  2::2,  2::2] += 0.125*uc
        uf[:-1:2,  1::2,  2::2,  2::2] += 0.125*uc
        uf[ 1::2,  2::2, :-1:2, :-1:2] += 0.125*uc
        uf[ 1::2, :-1:2, :-1:2, :-1:2] += 0.125*uc
        uf[ 1::2,  2::2,  2::2, :-1:2] += 0.125*uc
        uf[ 1::2, :-1:2,  2::2, :-1:2] += 0.125*uc
        uf[ 1::2,  2::2, :-1:2,  2::2] += 0.125*uc
        uf[ 1::2, :-1:2, :-1:2,  2::2] += 0.125*uc
        uf[ 1::2,  2::2,  2::2,  2::2] += 0.125*uc
        uf[ 1::2, :-1:2,  2::2,  2::2] += 0.125*uc

        uf[ 2::2,  2::2, :-1:2, :-1:2] += 0.0625*uc
        uf[:-1:2,  2::2, :-1:2, :-1:2] += 0.0625*uc
        uf[ 2::2, :-1:2, :-1:2, :-1:2] += 0.0625*uc
        uf[:-1:2, :-1:2, :-1:2, :-1:2] += 0.0625*uc
        uf[ 2::2,  2::2,  2::2, :-1:2] += 0.0625*uc
        uf[:-1:2,  2::2,  2::2, :-1:2] += 0.0625*uc
        uf[ 2::2, :-1:2,  2::2, :-1:2] += 0.0625*uc
        uf[:-1:2, :-1:2,  2::2, :-1:2] += 0.0625*uc
        uf[ 2::2,  2::2, :-1:2,  2::2] += 0.0625*uc
        uf[:-1:2,  2::2, :-1:2,  2::2] += 0.0625*uc
        uf[ 2::2, :-1:2, :-1:2,  2::2] += 0.0625*uc
        uf[:-1:2, :-1:2, :-1:2,  2::2] += 0.0625*uc
        uf[ 2::2,  2::2,  2::2,  2::2] += 0.0625*uc
        uf[:-1:2,  2::2,  2::2,  2::2] += 0.0625*uc
        uf[ 2::2, :-1:2,  2::2,  2::2] += 0.0625*uc
        uf[:-1:2, :-1:2,  2::2,  2::2] += 0.0625*uc
    else: raise ValueError(f"not written in dim={dim}")
    # return uf
def _restrictsimple(dim, uc, uf):
    if dim==1:
        uc += uf[ 1::2]
        uc += 0.5*uf[:-1:2]
        uc += 0.5*uf[ 2::2]
    elif dim==2:
        uc += uf[ 1::2,  1::2]
        uc += 0.5 *uf[:-1:2,  1::2]
        uc += 0.5 *uf[ 2::2,  1::2]
        uc += 0.5 *uf[ 1::2, :-1:2]
        uc += 0.5 *uf[ 1::2,  2::2]
        uc += 0.25*uf[:-1:2, :-1:2]
        uc += 0.25*uf[ 2::2, :-1:2]
        uc += 0.25*uf[:-1:2,  2::2]
        uc += 0.25*uf[ 2::2,  2::2]
    elif dim==3:
        uc += uf[1::2, 1::2, 1::2]
        uc += 0.5  *uf[:-1:2,  1::2,  1::2]
        uc += 0.5  *uf[ 2::2,  1::2,  1::2]
        uc += 0.5  *uf[ 1::2, :-1:2,  1::2]
        uc += 0.5  *uf[ 1::2,  2::2,  1::2]
        uc += 0.5  *uf[ 1::2,  1::2, :-1:2]
        uc += 0.5  *uf[ 1::2,  1::2,  2::2]
        uc += 0.25 *uf[:-1:2, :-1:2,  1::2]
        uc += 0.25 *uf[ 2::2, :-1:2,  1::2]
        uc += 0.25 *uf[:-1:2,  2::2,  1::2]
        uc += 0.25 *uf[ 2::2,  2::2,  1::2]
        uc += 0.25 *uf[:-1:2,  1::2, :-1:2]
        uc += 0.25 *uf[ 2::2,  1::2, :-1:2]
        uc += 0.25 *uf[:-1:2,  1::2,  2::2]
        uc += 0.25 *uf[ 2::2,  1::2,  2::2]
        uc += 0.25 *uf[ 1::2, :-1:2, :-1:2]
        uc += 0.25 *uf[ 1::2,  2::2, :-1:2]
        uc += 0.25 *uf[ 1::2, :-1:2,  2::2]
        uc += 0.25 *uf[ 1::2,  2::2,  2::2]
        uc += 0.125*uf[ 2::2, :-1:2, :-1:2]
        uc += 0.125*uf[:-1:2, :-1:2, :-1:2]
        uc += 0.125*uf[ 2::2,  2::2, :-1:2]
        uc += 0.125*uf[:-1:2,  2::2, :-1:2]
        uc += 0.125*uf[ 2::2, :-1:2,  2::2]
        uc += 0.125*uf[:-1:2, :-1:2,  2::2]
        uc += 0.125*uf[ 2::2,  2::2,  2::2]
        uc += 0.125*uf[:-1:2,  2::2,  2::2]
    elif dim==4:
        uc += uf[1::2, 1::2, 1::2, 1::2]
        uc += 0.5   *uf[:-1:2,  1::2,  1::2,  1::2]
        uc += 0.5   *uf[ 2::2,  1::2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2, :-1:2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2,  2::2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2, :-1:2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2,  2::2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2,  1::2,  2::2]
        uc += 0.5   *uf[ 1::2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[:-1:2, :-1:2,  1::2,  1::2]
        uc += 0.25  *uf[ 2::2, :-1:2,  1::2,  1::2]
        uc += 0.25  *uf[:-1:2,  2::2,  1::2,  1::2]
        uc += 0.25  *uf[ 2::2,  2::2,  1::2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2, :-1:2,  1::2]
        uc += 0.25  *uf[ 2::2,  1::2, :-1:2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2,  2::2,  1::2]
        uc += 0.25  *uf[ 2::2,  1::2,  2::2,  1::2]
        uc += 0.25  *uf[ 1::2, :-1:2, :-1:2,  1::2]
        uc += 0.25  *uf[ 1::2,  2::2, :-1:2,  1::2]
        uc += 0.25  *uf[ 1::2, :-1:2,  2::2,  1::2]
        uc += 0.25  *uf[ 1::2,  2::2,  2::2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2,  1::2,  2::2]
        uc += 0.25  *uf[ 2::2,  1::2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2, :-1:2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2,  2::2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2,  1::2, :-1:2,  2::2]
        uc += 0.25  *uf[ 1::2,  1::2,  2::2,  2::2]
        uc += 0.25  *uf[:-1:2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 2::2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2, :-1:2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2,  2::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2,  1::2, :-1:2, :-1:2]
        uc += 0.25  *uf[ 1::2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 2::2, :-1:2, :-1:2,  1::2]
        uc += 0.125 *uf[:-1:2, :-1:2, :-1:2,  1::2]
        uc += 0.125 *uf[ 2::2,  2::2, :-1:2,  1::2]
        uc += 0.125 *uf[:-1:2,  2::2, :-1:2,  1::2]
        uc += 0.125 *uf[ 2::2, :-1:2,  2::2,  1::2]
        uc += 0.125 *uf[:-1:2, :-1:2,  2::2,  1::2]
        uc += 0.125 *uf[ 2::2,  2::2,  2::2,  1::2]
        uc += 0.125 *uf[:-1:2,  2::2,  2::2,  1::2]
        uc += 0.125 *uf[ 2::2, :-1:2,  1::2, :-1:2]
        uc += 0.125 *uf[:-1:2, :-1:2,  1::2, :-1:2]
        uc += 0.125 *uf[ 2::2,  2::2,  1::2, :-1:2]
        uc += 0.125 *uf[:-1:2,  2::2,  1::2, :-1:2]
        uc += 0.125 *uf[ 2::2, :-1:2,  1::2,  2::2]
        uc += 0.125 *uf[:-1:2, :-1:2,  1::2,  2::2]
        uc += 0.125 *uf[ 2::2,  2::2,  1::2,  2::2]
        uc += 0.125 *uf[:-1:2,  2::2,  1::2,  2::2]
        uc += 0.125 *uf[ 2::2,  1::2, :-1:2, :-1:2]
        uc += 0.125 *uf[:-1:2,  1::2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 2::2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[:-1:2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 2::2,  1::2, :-1:2,  2::2]
        uc += 0.125 *uf[:-1:2,  1::2, :-1:2,  2::2]
        uc += 0.125 *uf[ 2::2,  1::2,  2::2,  2::2]
        uc += 0.125 *uf[:-1:2,  1::2,  2::2,  2::2]
        uc += 0.125 *uf[ 1::2,  2::2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 1::2, :-1:2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 1::2,  2::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 1::2, :-1:2,  2::2, :-1:2]
        uc += 0.125 *uf[ 1::2,  2::2, :-1:2,  2::2]
        uc += 0.125 *uf[ 1::2, :-1:2, :-1:2,  2::2]
        uc += 0.125 *uf[ 1::2,  2::2,  2::2,  2::2]
        uc += 0.125 *uf[ 1::2, :-1:2,  2::2,  2::2]
        uc += 0.0625*uf[ 2::2,  2::2, :-1:2, :-1:2]
        uc += 0.0625*uf[:-1:2,  2::2, :-1:2, :-1:2]
        uc += 0.0625*uf[ 2::2, :-1:2, :-1:2, :-1:2]
        uc += 0.0625*uf[:-1:2, :-1:2, :-1:2, :-1:2]
        uc += 0.0625*uf[ 2::2,  2::2,  2::2, :-1:2]
        uc += 0.0625*uf[:-1:2,  2::2,  2::2, :-1:2]
        uc += 0.0625*uf[ 2::2, :-1:2,  2::2, :-1:2]
        uc += 0.0625*uf[:-1:2, :-1:2,  2::2, :-1:2]
        uc += 0.0625*uf[ 2::2,  2::2, :-1:2,  2::2]
        uc += 0.0625*uf[:-1:2,  2::2, :-1:2,  2::2]
        uc += 0.0625*uf[ 2::2, :-1:2, :-1:2,  2::2]
        uc += 0.0625*uf[:-1:2, :-1:2, :-1:2,  2::2]
        uc += 0.0625*uf[ 2::2,  2::2,  2::2,  2::2]
        uc += 0.0625*uf[:-1:2,  2::2,  2::2,  2::2]
        uc += 0.0625*uf[ 2::2, :-1:2,  2::2,  2::2]
        uc += 0.0625*uf[:-1:2, :-1:2,  2::2,  2::2]
    else: raise ValueError(f"not written in dim={dim}")
    # return uc
#-----------------------------------------------------------------#
def _interpolatesimplenumba(dim, uf, uc):
    if   dim==1: _interpolatesimplenumba1D(uf, uc)
    elif dim==2: _interpolatesimplenumba2D(uf, uc)
    elif dim==3: _interpolatesimplenumba3D(uf, uc)
    elif dim==4: _interpolatesimplenumba4D(uf, uc)
    else: raise ValueError(f"not written in dim={dim}")
# @jit(nopython=True)
def _interpolatesimplenumba1D(uf, uc):
    uf[1::2] += uc
    uf[:-1:2] += 0.5 * uc
    uf[2::2] += 0.5 * uc
# @jit(nopython=True)
def _interpolatesimplenumba2D(uf, uc):
    uf[1::2, 1::2] += uc
    uf[:-1:2, 1::2] += 0.5 * uc
    uf[2::2, 1::2] += 0.5 * uc
    uf[1::2, :-1:2] += 0.5 * uc
    uf[1::2, 2::2] += 0.5 * uc
    uf[:-1:2, :-1:2] += 0.25 * uc
    uf[2::2, :-1:2] += 0.25 * uc
    uf[:-1:2, 2::2] += 0.25 * uc
    uf[2::2, 2::2] += 0.25 * uc
# @jit(nopython=True)
def _interpolatesimplenumba3D(uf, uc):
    uf[1::2, 1::2, 1::2] += uc
    uf[:-1:2, 1::2, 1::2] += 0.5 * uc
    uf[2::2, 1::2, 1::2] += 0.5 * uc
    uf[1::2, :-1:2, 1::2] += 0.5 * uc
    uf[1::2, 2::2, 1::2] += 0.5 * uc
    uf[1::2, 1::2, :-1:2] += 0.5 * uc
    uf[1::2, 1::2, 2::2] += 0.5 * uc
    uf[:-1:2, :-1:2, 1::2] += 0.25 * uc
    uf[2::2, :-1:2, 1::2] += 0.25 * uc
    uf[:-1:2, 2::2, 1::2] += 0.25 * uc
    uf[2::2, 2::2, 1::2] += 0.25 * uc
    uf[:-1:2, 1::2, :-1:2] += 0.25 * uc
    uf[2::2, 1::2, :-1:2] += 0.25 * uc
    uf[:-1:2, 1::2, 2::2] += 0.25 * uc
    uf[2::2, 1::2, 2::2] += 0.25 * uc
    uf[1::2, :-1:2, :-1:2] += 0.25 * uc
    uf[1::2, 2::2, :-1:2] += 0.25 * uc
    uf[1::2, :-1:2, 2::2] += 0.25 * uc
    uf[1::2, 2::2, 2::2] += 0.25 * uc
    uf[2::2, :-1:2, :-1:2] += 0.125 * uc
    uf[:-1:2, :-1:2, :-1:2] += 0.125 * uc
    uf[2::2, 2::2, :-1:2] += 0.125 * uc
    uf[:-1:2, 2::2, :-1:2] += 0.125 * uc
    uf[2::2, :-1:2, 2::2] += 0.125 * uc
    uf[:-1:2, :-1:2, 2::2] += 0.125 * uc
    uf[2::2, 2::2, 2::2] += 0.125 * uc
    uf[:-1:2, 2::2, 2::2] += 0.125 * uc
# @jit(nopython=True)
def _interpolatesimplenumba4D(uf, uc):
    uf[1::2, 1::2, 1::2, 1::2] += uc
    uf[:-1:2, 1::2, 1::2, 1::2] += 0.5 * uc
    uf[2::2, 1::2, 1::2, 1::2] += 0.5 * uc
    uf[1::2, :-1:2, 1::2, 1::2] += 0.5 * uc
    uf[1::2, 2::2, 1::2, 1::2] += 0.5 * uc
    uf[1::2, 1::2, :-1:2, 1::2] += 0.5 * uc
    uf[1::2, 1::2, 2::2, 1::2] += 0.5 * uc
    uf[1::2, 1::2, 1::2, 2::2] += 0.5 * uc
    uf[1::2, 1::2, 1::2, :-1:2] += 0.5 * uc

    uf[:-1:2, :-1:2, 1::2, 1::2] += 0.25 * uc
    uf[2::2, :-1:2, 1::2, 1::2] += 0.25 * uc
    uf[:-1:2, 2::2, 1::2, 1::2] += 0.25 * uc
    uf[2::2, 2::2, 1::2, 1::2] += 0.25 * uc
    uf[:-1:2, 1::2, :-1:2, 1::2] += 0.25 * uc
    uf[2::2, 1::2, :-1:2, 1::2] += 0.25 * uc
    uf[:-1:2, 1::2, 2::2, 1::2] += 0.25 * uc
    uf[2::2, 1::2, 2::2, 1::2] += 0.25 * uc
    uf[1::2, :-1:2, :-1:2, 1::2] += 0.25 * uc
    uf[1::2, 2::2, :-1:2, 1::2] += 0.25 * uc
    uf[1::2, :-1:2, 2::2, 1::2] += 0.25 * uc
    uf[1::2, 2::2, 2::2, 1::2] += 0.25 * uc
    uf[:-1:2, 1::2, 1::2, 2::2] += 0.25 * uc
    uf[2::2, 1::2, 1::2, 2::2] += 0.25 * uc
    uf[1::2, :-1:2, 1::2, 2::2] += 0.25 * uc
    uf[1::2, 2::2, 1::2, 2::2] += 0.25 * uc
    uf[1::2, 1::2, :-1:2, 2::2] += 0.25 * uc
    uf[1::2, 1::2, 2::2, 2::2] += 0.25 * uc
    uf[:-1:2, 1::2, 1::2, :-1:2] += 0.25 * uc
    uf[2::2, 1::2, 1::2, :-1:2] += 0.25 * uc
    uf[1::2, :-1:2, 1::2, :-1:2] += 0.25 * uc
    uf[1::2, 2::2, 1::2, :-1:2] += 0.25 * uc
    uf[1::2, 1::2, :-1:2, :-1:2] += 0.25 * uc
    uf[1::2, 1::2, 2::2, :-1:2] += 0.25 * uc

    uf[2::2, :-1:2, :-1:2, 1::2] += 0.125 * uc
    uf[:-1:2, :-1:2, :-1:2, 1::2] += 0.125 * uc
    uf[2::2, 2::2, :-1:2, 1::2] += 0.125 * uc
    uf[:-1:2, 2::2, :-1:2, 1::2] += 0.125 * uc
    uf[2::2, :-1:2, 2::2, 1::2] += 0.125 * uc
    uf[:-1:2, :-1:2, 2::2, 1::2] += 0.125 * uc
    uf[2::2, 2::2, 2::2, 1::2] += 0.125 * uc
    uf[:-1:2, 2::2, 2::2, 1::2] += 0.125 * uc
    uf[2::2, :-1:2, 1::2, :-1:2] += 0.125 * uc
    uf[:-1:2, :-1:2, 1::2, :-1:2] += 0.125 * uc
    uf[2::2, 2::2, 1::2, :-1:2] += 0.125 * uc
    uf[:-1:2, 2::2, 1::2, :-1:2] += 0.125 * uc
    uf[2::2, :-1:2, 1::2, 2::2] += 0.125 * uc
    uf[:-1:2, :-1:2, 1::2, 2::2] += 0.125 * uc
    uf[2::2, 2::2, 1::2, 2::2] += 0.125 * uc
    uf[:-1:2, 2::2, 1::2, 2::2] += 0.125 * uc
    uf[2::2, 1::2, :-1:2, :-1:2] += 0.125 * uc
    uf[:-1:2, 1::2, :-1:2, :-1:2] += 0.125 * uc
    uf[2::2, 1::2, 2::2, :-1:2] += 0.125 * uc
    uf[:-1:2, 1::2, 2::2, :-1:2] += 0.125 * uc
    uf[2::2, 1::2, :-1:2, 2::2] += 0.125 * uc
    uf[:-1:2, 1::2, :-1:2, 2::2] += 0.125 * uc
    uf[2::2, 1::2, 2::2, 2::2] += 0.125 * uc
    uf[:-1:2, 1::2, 2::2, 2::2] += 0.125 * uc
    uf[1::2, 2::2, :-1:2, :-1:2] += 0.125 * uc
    uf[1::2, :-1:2, :-1:2, :-1:2] += 0.125 * uc
    uf[1::2, 2::2, 2::2, :-1:2] += 0.125 * uc
    uf[1::2, :-1:2, 2::2, :-1:2] += 0.125 * uc
    uf[1::2, 2::2, :-1:2, 2::2] += 0.125 * uc
    uf[1::2, :-1:2, :-1:2, 2::2] += 0.125 * uc
    uf[1::2, 2::2, 2::2, 2::2] += 0.125 * uc
    uf[1::2, :-1:2, 2::2, 2::2] += 0.125 * uc

    uf[2::2, 2::2, :-1:2, :-1:2] += 0.0625 * uc
    uf[:-1:2, 2::2, :-1:2, :-1:2] += 0.0625 * uc
    uf[2::2, :-1:2, :-1:2, :-1:2] += 0.0625 * uc
    uf[:-1:2, :-1:2, :-1:2, :-1:2] += 0.0625 * uc
    uf[2::2, 2::2, 2::2, :-1:2] += 0.0625 * uc
    uf[:-1:2, 2::2, 2::2, :-1:2] += 0.0625 * uc
    uf[2::2, :-1:2, 2::2, :-1:2] += 0.0625 * uc
    uf[:-1:2, :-1:2, 2::2, :-1:2] += 0.0625 * uc
    uf[2::2, 2::2, :-1:2, 2::2] += 0.0625 * uc
    uf[:-1:2, 2::2, :-1:2, 2::2] += 0.0625 * uc
    uf[2::2, :-1:2, :-1:2, 2::2] += 0.0625 * uc
    uf[:-1:2, :-1:2, :-1:2, 2::2] += 0.0625 * uc
    uf[2::2, 2::2, 2::2, 2::2] += 0.0625 * uc
    uf[:-1:2, 2::2, 2::2, 2::2] += 0.0625 * uc
    uf[2::2, :-1:2, 2::2, 2::2] += 0.0625 * uc
    uf[:-1:2, :-1:2, 2::2, 2::2] += 0.0625 * uc


def _restrictsimplenumba(dim, uc, uf):
    if dim==1:
        uc += uf[ 1::2]
        uc += 0.5*uf[:-1:2]
        uc += 0.5*uf[ 2::2]
    elif dim==2:
        uc += uf[ 1::2,  1::2]
        uc += 0.5 *uf[:-1:2,  1::2]
        uc += 0.5 *uf[ 2::2,  1::2]
        uc += 0.5 *uf[ 1::2, :-1:2]
        uc += 0.5 *uf[ 1::2,  2::2]
        uc += 0.25*uf[:-1:2, :-1:2]
        uc += 0.25*uf[ 2::2, :-1:2]
        uc += 0.25*uf[:-1:2,  2::2]
        uc += 0.25*uf[ 2::2,  2::2]
    elif dim==3:
        uc += uf[1::2, 1::2, 1::2]
        uc += 0.5  *uf[:-1:2,  1::2,  1::2]
        uc += 0.5  *uf[ 2::2,  1::2,  1::2]
        uc += 0.5  *uf[ 1::2, :-1:2,  1::2]
        uc += 0.5  *uf[ 1::2,  2::2,  1::2]
        uc += 0.5  *uf[ 1::2,  1::2, :-1:2]
        uc += 0.5  *uf[ 1::2,  1::2,  2::2]
        uc += 0.25 *uf[:-1:2, :-1:2,  1::2]
        uc += 0.25 *uf[ 2::2, :-1:2,  1::2]
        uc += 0.25 *uf[:-1:2,  2::2,  1::2]
        uc += 0.25 *uf[ 2::2,  2::2,  1::2]
        uc += 0.25 *uf[:-1:2,  1::2, :-1:2]
        uc += 0.25 *uf[ 2::2,  1::2, :-1:2]
        uc += 0.25 *uf[:-1:2,  1::2,  2::2]
        uc += 0.25 *uf[ 2::2,  1::2,  2::2]
        uc += 0.25 *uf[ 1::2, :-1:2, :-1:2]
        uc += 0.25 *uf[ 1::2,  2::2, :-1:2]
        uc += 0.25 *uf[ 1::2, :-1:2,  2::2]
        uc += 0.25 *uf[ 1::2,  2::2,  2::2]
        uc += 0.125*uf[ 2::2, :-1:2, :-1:2]
        uc += 0.125*uf[:-1:2, :-1:2, :-1:2]
        uc += 0.125*uf[ 2::2,  2::2, :-1:2]
        uc += 0.125*uf[:-1:2,  2::2, :-1:2]
        uc += 0.125*uf[ 2::2, :-1:2,  2::2]
        uc += 0.125*uf[:-1:2, :-1:2,  2::2]
        uc += 0.125*uf[ 2::2,  2::2,  2::2]
        uc += 0.125*uf[:-1:2,  2::2,  2::2]
    elif dim==4:
        uc += uf[1::2, 1::2, 1::2, 1::2]
        uc += 0.5   *uf[:-1:2,  1::2,  1::2,  1::2]
        uc += 0.5   *uf[ 2::2,  1::2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2, :-1:2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2,  2::2,  1::2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2, :-1:2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2,  2::2,  1::2]
        uc += 0.5   *uf[ 1::2,  1::2,  1::2,  2::2]
        uc += 0.5   *uf[ 1::2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[:-1:2, :-1:2,  1::2,  1::2]
        uc += 0.25  *uf[ 2::2, :-1:2,  1::2,  1::2]
        uc += 0.25  *uf[:-1:2,  2::2,  1::2,  1::2]
        uc += 0.25  *uf[ 2::2,  2::2,  1::2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2, :-1:2,  1::2]
        uc += 0.25  *uf[ 2::2,  1::2, :-1:2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2,  2::2,  1::2]
        uc += 0.25  *uf[ 2::2,  1::2,  2::2,  1::2]
        uc += 0.25  *uf[ 1::2, :-1:2, :-1:2,  1::2]
        uc += 0.25  *uf[ 1::2,  2::2, :-1:2,  1::2]
        uc += 0.25  *uf[ 1::2, :-1:2,  2::2,  1::2]
        uc += 0.25  *uf[ 1::2,  2::2,  2::2,  1::2]
        uc += 0.25  *uf[:-1:2,  1::2,  1::2,  2::2]
        uc += 0.25  *uf[ 2::2,  1::2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2, :-1:2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2,  2::2,  1::2,  2::2]
        uc += 0.25  *uf[ 1::2,  1::2, :-1:2,  2::2]
        uc += 0.25  *uf[ 1::2,  1::2,  2::2,  2::2]
        uc += 0.25  *uf[:-1:2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 2::2,  1::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2, :-1:2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2,  2::2,  1::2, :-1:2]
        uc += 0.25  *uf[ 1::2,  1::2, :-1:2, :-1:2]
        uc += 0.25  *uf[ 1::2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 2::2, :-1:2, :-1:2,  1::2]
        uc += 0.125 *uf[:-1:2, :-1:2, :-1:2,  1::2]
        uc += 0.125 *uf[ 2::2,  2::2, :-1:2,  1::2]
        uc += 0.125 *uf[:-1:2,  2::2, :-1:2,  1::2]
        uc += 0.125 *uf[ 2::2, :-1:2,  2::2,  1::2]
        uc += 0.125 *uf[:-1:2, :-1:2,  2::2,  1::2]
        uc += 0.125 *uf[ 2::2,  2::2,  2::2,  1::2]
        uc += 0.125 *uf[:-1:2,  2::2,  2::2,  1::2]
        uc += 0.125 *uf[ 2::2, :-1:2,  1::2, :-1:2]
        uc += 0.125 *uf[:-1:2, :-1:2,  1::2, :-1:2]
        uc += 0.125 *uf[ 2::2,  2::2,  1::2, :-1:2]
        uc += 0.125 *uf[:-1:2,  2::2,  1::2, :-1:2]
        uc += 0.125 *uf[ 2::2, :-1:2,  1::2,  2::2]
        uc += 0.125 *uf[:-1:2, :-1:2,  1::2,  2::2]
        uc += 0.125 *uf[ 2::2,  2::2,  1::2,  2::2]
        uc += 0.125 *uf[:-1:2,  2::2,  1::2,  2::2]
        uc += 0.125 *uf[ 2::2,  1::2, :-1:2, :-1:2]
        uc += 0.125 *uf[:-1:2,  1::2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 2::2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[:-1:2,  1::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 2::2,  1::2, :-1:2,  2::2]
        uc += 0.125 *uf[:-1:2,  1::2, :-1:2,  2::2]
        uc += 0.125 *uf[ 2::2,  1::2,  2::2,  2::2]
        uc += 0.125 *uf[:-1:2,  1::2,  2::2,  2::2]
        uc += 0.125 *uf[ 1::2,  2::2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 1::2, :-1:2, :-1:2, :-1:2]
        uc += 0.125 *uf[ 1::2,  2::2,  2::2, :-1:2]
        uc += 0.125 *uf[ 1::2, :-1:2,  2::2, :-1:2]
        uc += 0.125 *uf[ 1::2,  2::2, :-1:2,  2::2]
        uc += 0.125 *uf[ 1::2, :-1:2, :-1:2,  2::2]
        uc += 0.125 *uf[ 1::2,  2::2,  2::2,  2::2]
        uc += 0.125 *uf[ 1::2, :-1:2,  2::2,  2::2]
        uc += 0.0625*uf[ 2::2,  2::2, :-1:2, :-1:2]
        uc += 0.0625*uf[:-1:2,  2::2, :-1:2, :-1:2]
        uc += 0.0625*uf[ 2::2, :-1:2, :-1:2, :-1:2]
        uc += 0.0625*uf[:-1:2, :-1:2, :-1:2, :-1:2]
        uc += 0.0625*uf[ 2::2,  2::2,  2::2, :-1:2]
        uc += 0.0625*uf[:-1:2,  2::2,  2::2, :-1:2]
        uc += 0.0625*uf[ 2::2, :-1:2,  2::2, :-1:2]
        uc += 0.0625*uf[:-1:2, :-1:2,  2::2, :-1:2]
        uc += 0.0625*uf[ 2::2,  2::2, :-1:2,  2::2]
        uc += 0.0625*uf[:-1:2,  2::2, :-1:2,  2::2]
        uc += 0.0625*uf[ 2::2, :-1:2, :-1:2,  2::2]
        uc += 0.0625*uf[:-1:2, :-1:2, :-1:2,  2::2]
        uc += 0.0625*uf[ 2::2,  2::2,  2::2,  2::2]
        uc += 0.0625*uf[:-1:2,  2::2,  2::2,  2::2]
        uc += 0.0625*uf[ 2::2, :-1:2,  2::2,  2::2]
        uc += 0.0625*uf[:-1:2, :-1:2,  2::2,  2::2]
    else: raise ValueError(f"not written in dim={dim}")
#-----------------------------------------------------------------#
def interpolateMat(gridf, gridc):
    print(f"gridf = {gridf.n} gridc = {gridc.n}")
    diags = np.ones((3,3,gridf.nall()))
    diagonals = []
    offsets = []
    dim, nallc, nallf = gridc.dim, gridc.nall(), gridf.nall()
    stridesc, stridesf = gridc.strides(), gridf.strides()
    assert dim==2
    for ii in range(3):
        sti = [-stridesc[0], 0, stridesc[0]]
        for jj in range(3):
            stj = [-stridesc[1], 0, stridesc[1]]
            stride = sti[ii] + stj[jj]
            if ii == 1 and jj == 1:
                if stride <=0: diagonals.append(diags[ii,jj,-stride:])
                else: diagonals.append(diags[ii,jj,:stride])
                offsets.append(stride)
    print(f"offsets={offsets} stridesc={stridesc}")
    A = sp.sparse.diags(diagonals=diagonals, offsets=offsets, shape=(nallc,nallf))
    np.savetxt(f"transfermatrix.txt", A.toarray(), fmt="%6.2f")
    # print("A=\n", A.toarray())
    # return matrix.createMatrix3pd(gridc, diags)
def interpolateWithMat(gridf, gridc, uc, transpose=False, unew=None):
    A = interpolateMat(gridf, gridc)
    return A.T.dot(uc)
#-----------------------------------------------------------------#
def interpolate(gridf, gridc, uold, transpose=False, unew=None, level=None):
    nold = gridc.n
    uold = uold.ravel()
    if transpose:
        if unew is None: unew = np.zeros(gridc.nall())
        else: unew.shape = gridc.nall()
        if uold.shape[0] != gridf.nall():
            raise ValueError(f"Problem interpolate(transpose={transpose}) {uc.shape[0]} != {gridf.nall()} ({gridc.nall()})")
    else:
        if unew is None: unew = np.zeros(gridf.nall())
        else: unew.shape = gridf.nall()
        if uold.shape[0] != gridc.nall():
            raise ValueError(f"Problem interpolate(transpose={transpose}) u.N = {uc.shape[0]} != {gridc.nall()} = grid.N")
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
#-----------------------------------------------------------------#
def error(g, u, uex):
    err = np.linalg.norm(u.reshape(g.n) - uex(*g.coord())) / np.sqrt(g.nall())
    if err > 1e-15:
        print(f"err={err:12.4e}\n uex=\n{uex}")
        print(f"uc={u.sum()}\n u=\n{uex(g.coord()).sum()}")
        raise ValueError(f"err = {err}")
    return err
def testprolongation(ns, bounds, expr, fcts, transpose=False):
    import time
    d = len(bounds)
    uex = anasol.AnalyticalSolution(expr, dim=d)
    print(f"uex={uex}")
    grids = []
    for n in ns:
        grids.append(grid.Grid(n, bounds=bounds))
    intsimp = InterpolateSimple(grids)
    times = {}
    for fct in fcts: times[fct] = []
    N = []
    if transpose:
        args = "(gridf = gf, gridc = gc, uold=uc, level=l+1)"
        argsT = "(gridf = gf, gridc = gc, uold=uf, level=l+1, transpose=True)"
        for l in range(len(grids)-1):
            gc = grids[l]
            gf = grids[l+1]
            N.append(gf.nall())
            uf = np.random.random(gf.nall())
            uc = np.random.random(gc.nall())
            uc = np.linspace(0, 1, gc.nall())
            uf = np.ones(gf.nall())
            for fct in fcts:
                print(fct)
                t0 = time.time()
                # uf2 = eval(fct+args)
                # print(f"AVANT uf2 = {uf2} ({id(uf2)}) {adress(uf2)}")
                uc2 = eval(fct+argsT)
                # print(f"uc = {uc}\nuc2 = {uc2} ({id(uc2)})")
                # print(f"uc={uc.shape} uc2={uc2.shape}")
                uf2 = eval(fct+args)
                # print(f"APRES uf2 = {uf2} ({id(uf2)}) {adress(uf2)}")
                times[fct].append(time.time()-t0)
                # print(f"uc={uc.shape} uc2={uc2.shape}")
                suc, suf = uc2.dot(uc),  uf2.dot(uf)
                if abs(suc-suf) > 1e-14*uc.dot(uc):
                    print(f"uc2*uc = {suc} uf2*uf = {suf}")
                    raise ValueError(f"bug in restrict {abs(suc-suf)}")
    else:
        args = "(gridf = gf, gridc = gc, uold=uc, level=l+1)"
        uc = uex(*grids[0].coord())
        for l in range(len(grids)-1):
            gc = grids[l]
            gf = grids[l+1]
            N.append(gf.nall())
            for fct in fcts:
                print(fct)
                t0 = time.time()
                uf  = eval(fct+args)
                times[fct].append(time.time()-t0)
                error(gf, uf, uex)
            uc = uf

    for fct in fcts:
        plt.plot(np.log10(N), np.log10(times[fct]), 'x-', label=fct)
    plt.title(f"d={d}")
    plt.legend()
    plt.xlabel("log10(n)")
    plt.ylabel("log10(t)")
    plt.show()
#-----------------------------------------------------------------#
def testadjoint():
    d = 2
    gc = grid.Grid(d * [3], bounds=d * [[-1, 1]])
    gf = grid.Grid(d * [5], bounds=d * [[-1, 1]])
    uf = np.random.random(gf.nall())
    uc = np.random.random(gc.nall())
    uc2 = interpolate(gf, gc, uf, transpose=True)
    uf2 = interpolate(gf, gc, uc)
    print(f"uc2*uc = {uc2.dot(uc)} uf2*uf = {uf2.dot(uf)}")
#-----------------------------------------------------------------#
def testtocell(d=1):
    g = grid.Grid(d * [3], bounds=d * [[-1, 1]])
    expr = ''
    for i in range(d): expr += f"{np.random.randint(low=1,high=9)}*x{i}+"
    uex = anasol.AnalyticalSolution(expr[:-1], dim=d)
    u = uex(g.coord())
    # print(f"grid = {g}\nuex={uex}\nu={u}")
    uc = tocell(g, u)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plotgrid.plot(g, ax=ax, u=u, title=f"d={d}")
    ax = fig.add_subplot(2, 1, 2)
    plotgrid.plot(g, ax=ax, u=uc, title=f"d={d}", celldata=True)
    plt.show()

#=================================================================#
if __name__ == '__main__':
    # testadjoint()
    # testtocell(d=2)

    test1d, test2d, test3d, test4d = True, True, True, True
    # test1d, test2d, test3d, test4d = True, False, False, False
    transpose = False
    fcts = ['interpolate', 'interpolateWithMat']
    fcts = ['interpolate', 'interpolatesimple', 'interpolatenumba']
    fcts = ['interpolate', 'interpolatesimple', 'intsimp.interpolate']
    fcts = ['interpolate', 'interpolatesimple', 'interpolatesimplenumba']
    if test1d:
        ns = [np.array([2])]
        for k in range(10): ns.append(2*ns[k]-1)
        expr = 'pi+pi*pi*x'
        # testprolongation(ns, bounds=[[-1,1]], expr=expr)
        testprolongation(ns, bounds=[[-1, 1]], expr=expr, fcts=fcts, transpose=transpose)
    if test2d:
        ns = [np.array([2,3])]
        for k in range(10): ns.append(2*ns[k]-1)
        expr = 'x+pi*y + 7*x*y'
        testprolongation(ns, bounds=2*[[-1,1]], expr=expr, fcts=fcts, transpose=transpose)
    if test3d:
        ns = [np.array([2,3,2])]
        for k in range(6): ns.append(2*ns[k]-1)
        expr = 'x+2*y+3*z-x*y-pi*x*z + pi**2*y*z'
        testprolongation(ns, bounds=3*[[-1,1]], expr=expr, fcts=fcts, transpose=transpose)
    if test4d:
        ns = [np.array([2,2,2,3])]
        for k in range(5): ns.append(2*ns[k]-1)
        expr = 'x0+2*x1+3*x2+4*x3-x0*x2-pi*x3*x1 + pi**2*x0*x3'
        testprolongation(ns, bounds=4*[[-1,1]], expr=expr, fcts=fcts, transpose=transpose)
