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
    d = grid.dim
    if ax == None:
        if d==3: ax = plt.gca(projection='3d')
        else: ax = plt.gca()
    if celldata: x = grid.coordCenters()
    else: x = grid.coord()
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
            ax.plot(x[0], x[1], marker='x', color='r', linestyle='none')
            cnt = ax.contour(*x, u)
            ax.clabel(cnt, cnt.levels, inline=True, fmt='%.1f', fontsize=10)
    elif d==3:
        if u is None:
            ax.scatter(x[0], x[1], x[2], color='r')
        else:
            from skimage.measure import marching_cubes_lewiner
            verts, faces, _, _ = marching_cubes_lewiner(u, np.mean(u), spacing=(0.1, 0.1, 0.1))
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)
        # else:
        #     import pyvista as pv
        #     grid = pv.StructuredGrid(x[0], x[1], x[2])
        #     grid["U"] = u
        #     contours = grid.contour([np.mean(u)])
        #     pv.set_plot_theme('document')
        #     p = pv.Plotter()
        #     p.add_mesh(contours, scalars=contours.points[:, 2], show_scalar_bar=False)
        #     p.show()
    else:
        raise ValueError(f"Not written: plot in d={d}")

    if not 'ax' in kwargs: plt.show()
#=================================================================#
if __name__ == '__main__':
    from strucfem import grid

    grid1 = grid.Grid(n=[5], bounds=[[1, 3]])
    plot(grid1, title="1D")
