import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------#
def plot(grid, **kwargs):
    """
    """
    u, ucft, ax, plot_error, title, celldata = None, None, None, False, "", False
    if 'u' in kwargs: u = kwargs.pop('u')
    if 'ufct' in kwargs: ufct = kwargs.pop('ufct')
    if 'plot_error' in kwargs: plot_error = kwargs.pop('plot_error')
    if 'title' in kwargs: title = kwargs.pop('title')
    if 'ax' in kwargs: ax = kwargs['ax']
    if 'celldata' in kwargs: celldata = kwargs.pop('celldata')
    if ax == None: ax = plt.gca()
    if celldata: x = grid.coordCenters()
    else: x = grid.coord()
    d = grid.dim
    ax.set_title(title)
    u = u.reshape(x[0].shape)
    if x[0].shape != u.shape:
        raise ValueError(f"Problem in plot x[0].shape = {x[0].shape} != {u.shape} = u.shape (celldata={celldata})")
    if d==1:
        if u is None:
            ax.plot(x[0], np.full_like(x[0], 1), 'rx-')
        else:
            ax.plot(x[0], u, '-xb')
        if plot_error:
            if ufct is None: raise ValueError("Problem: need exact solution")
            plt.plot(x.ravel(), ufct(x), '-r')
    elif d==2:
        if u is None:
            ax.plot(x[0], x[1], marker='x', color='r', linestyle='none')
        else:
            cnt = ax.contour(*x, u)
            ax.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
    elif d==3:
        if u is None:
            ax.scatter(x[0], x[1], x[2], color='r')
        else:
            raise ValueError(f"Problem not written")
    else:
        raise ValueError(f"Problem not written: plot in d={d}")

    if not 'ax' in kwargs: plt.show()
#=================================================================#
if __name__ == '__main__':
    from strucfem import grid

    grid1 = grid.Grid(n=[5], bounds=[[1, 3]])
    plot(grid1, title="1D")
