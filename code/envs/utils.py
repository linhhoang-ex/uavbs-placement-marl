import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple
import os


def save_img_from_rgba_arr(
    rgba_arr: np.ndarray, fpath: str = 'tmp/tmp.png',
    figsize: Tuple = (8, 8), dpi: int = 8, transparent=True
) -> None:
    '''Save RGBA array (n_rows, n_cols, 4) to an image without padding.
    Output image size = figsize * dpi - 10% of matplotlib's default padding'''
    plt.figure(figsize=figsize)
    plt.imshow(rgba_arr)        # vmin=0, vmax=1
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=transparent)
    plt.close()
    print(f"Image saved to {fpath}")

    from PIL import Image
    img = Image.open(fpath)
    img.show()


def create_cmap_alpha(cmap_name) -> None:
    '''Create a custom cmap in matplotlib with transperancy.
    Ref: https://stackoverflow.com/questions/51601272/python-matplotlib-heatmap-colorbar-from-transparent'''
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap_name)(range(ncolors))
    # change alpha values
    # color_array[:,-1] = np.linspace(0.0,1.0,ncolors)
    color_array[0, -1] = 0
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=f'{cmap_name}_alpha', colors=color_array)
    # unregister the old one (if exist) and register this new colormap in matplotlib
    matplotlib.cm.unregister_cmap(name=f'{cmap_name}_alpha')
    plt.colormaps.register(cmap=map_object)


# Create custom color maps with transperancy for generating heatmaps
cmaps = {'user': 'Blues', 'mbs': 'Greys', 'uav': 'Greens', 'self': 'Reds'}
for cmap in cmaps.values():
    create_cmap_alpha(cmap)


def gen_hist2d(
    locs: Dict[str, np.ndarray],    # locations of users, macro BSs, and drone BSs
    cmaps: Dict[str, str] = cmaps,  # color map to plot the histogram
    bound: float = 1000,            # boundary of the area, default = 1.5 km
    grid: float = 20,               # grid size of the heatmap, default = 20 m
    figsize: Tuple = (8, 8),        # image size in inches
    dpi: float = 8,               # dots per inch (dpi), output image ~ figsize * dpi
) -> np.ndarray:                    # the heatmap of all network entities
    '''(v1.3) Convert locations of all netowrk entities (users, macro and drone BSs)
    into a 2D histogram map, viewed as an RGBA array of shape (H, W, 4).
    - locs = dict{'user': np.ndarray of shape (2, n_users),
                'mbs': np.ndarray of shape (2, n_mbss),
                'uav': np.ndarray of shape (2, n_uavs),
                'self': np.ndarray of shape (2, n_uavs)}
    - output heatmap: width = height = figsize * dpi - 10% of matplotlib's default padding

    References:
    - https://stackoverflow.com/a/21940031/16467215
    - https://stackoverflow.com/a/67823421/16467215
    '''
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Generate 2D histogram image
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    for type, coordinates in locs.items():
        grid_ = grid if type == 'user' else 1.5 * grid
        ax.hist2d(x=coordinates[0, :], y=coordinates[1, :],
                  bins=[np.arange(-bound - 50, bound + 50, grid_),
                        np.arange(-bound - 50, bound + 50, grid_)],
                  cmap=f"{cmaps[type]}_alpha")    # plt.cm.Greys, 'Greys', 'Reds', 'Blues'
    ax.axis("off")              # turns off axes
    ax.margins(0, 0)            # turn of margins
    for item in [fig, ax]:      # turn of facecolor
        item.patch.set_facecolor('None')    # OR item.patch.set_visible(False)
    fig.tight_layout(pad=0)     # remove the padding (white space around the Axes)
    fig.canvas.draw()           # draw the canvas, cache the renderer
    rgba_arr = fig.canvas.buffer_rgba()     # shape = (H, W, 4)
    plt.close()

    '''Solution 2: using canvas.tostring_argb()'''
    # hm = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # ncols, nrows = fig.canvas.get_width_height()
    # hm = hm.reshape(nrows, ncols, 4)                            # ARGB format
    # hm_rgba = np.ones_like(hm)
    # hm_rgba[:,:,:3] = hm[:,:,1:]; hm_rgba[:,:,3] = hm[:,:,0]    # convert to RGBA
    # plt.close()

    return np.asarray(rgba_arr)
