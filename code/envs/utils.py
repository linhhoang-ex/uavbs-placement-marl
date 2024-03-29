import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple
import os

rng = np.random.default_rng()


def save_img_from_rgba_arr(
    rgba_arr: np.ndarray, fpath: str = 'tmp/tmp.png',
    figsize: Tuple = (4, 4), dpi: int = 64, transparent=True, show=False
) -> None:
    '''Save RGBA array (n_rows, n_cols, 4) to an image without padding.
    Output image size = figsize * dpi - 10% of matplotlib's default padding'''
    plt.figure(figsize=figsize)
    plt.imshow(rgba_arr)        # vmin=0, vmax=1
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi, transparent=transparent)
    if show is True:
        plt.show()
    plt.close()
    print(f"Image saved to {os.path.join(os.getcwd(), fpath)}")

    # from PIL import Image
    # img = Image.open(fpath)
    # img.show()


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
    mode: str = "marl",             # ["sarl", "marl"]
    cmaps: Dict[str, str] = cmaps,  # color map to plot the histogram
    bound: float = 1000,            # boundary of the area, default = 1.5 km
    grid_size: float = 50,               # grid size of the heatmap, default = 20 m
    grid_norm: float = 5,           # normalization for # of users in a grid
    figsize: Tuple = (2, 2),        # image size in inches
    dpi: float = 32,               # dots per inch (dpi), output image ~ figsize * dpi
) -> np.ndarray:                    # the heatmap of all network entities
    '''(v1.3) Convert locations of all netowrk entities (users, macro and drone BSs)
    into a 2D histogram map, viewed as an RGBA array of shape (H, W, 4).

    Params
    ------
    locs = {'user': np.ndarray of shape (2, n_users),
            'mbs': np.ndarray of shape (2, n_mbss),    # required if mode="marl"
            'uav': np.ndarray of shape (2, n_uavs),    # required if mode="marl"
            'self': np.ndarray of shape (2, n_uavs)}
    mode: ["marl", "sarl"]
        "marl": multi-agent mode
        "sarl": single-agent mode

    Returns
    -------
    output heatmap: width = height = figsize * dpi - 10% of matplotlib's default padding

    References
    ----------
    - https://stackoverflow.com/a/21940031/16467215
    - https://stackoverflow.com/a/67823421/16467215
    '''
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # Generate 2D histogram image
    fig = plt.figure(figsize=figsize, dpi=dpi, num=1, clear=True)
    ax = plt.gca()
    if locs != {}:
        if mode == "marl":
            plt_order = ['user', 'mbs', 'uav', 'self']
        elif mode == "sarl":
            plt_order = ['user', 'self']
        # for type, coordinates in locs.items():
        for type in plt_order:
            coordinates = locs[type]
            if type == 'user':
                norm = matplotlib.colors.Normalize(vmin=0, vmax=grid_norm, clip=True)
                grid_ = grid_size
            else:
                norm = None
                grid_ = 2 * grid_size
            ax.hist2d(x=coordinates[0, :], y=coordinates[1, :],
                      bins=[np.arange(-bound, bound + grid_, grid_),
                            np.arange(-bound, bound + grid_, grid_)],
                      cmap=f"{cmaps[type]}_alpha", norm=norm)
    ax.axis("off")              # turns off axes
    ax.margins(0, 0)            # turn of margins
    for item in [fig, ax]:      # turn of facecolor
        item.patch.set_facecolor('None')    # OR item.patch.set_visible(False)
    fig.tight_layout(pad=0)     # remove the padding (white space around the Axes)
    fig.canvas.draw()           # draw the canvas, cache the renderer
    rgba_arr = fig.canvas.buffer_rgba()     # shape = (H, W, 4)
    # plt.cla()
    # plt.clf()
    # fig.clear()
    plt.close('all')

    '''Solution 2: using canvas.tostring_argb()'''
    # hm = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # ncols, nrows = fig.canvas.get_width_height()
    # hm = hm.reshape(nrows, ncols, 4)                            # ARGB format
    # hm_rgba = np.ones_like(hm)
    # hm_rgba[:,:,:3] = hm[:,:,1:]; hm_rgba[:,:,3] = hm[:,:,0]    # convert to RGBA
    # plt.close()

    return np.array(rgba_arr, copy=True)


def get_obs_flattened(
    locs: Dict[str, np.ndarray],    # locations of users, macro BSs, and drone BSs
    bound: float = 1000,            # boundary of the area, default = 1.5 km
    grid_size: float = 50,          # grid size of the heatmap, default = 20 m
    grid_norm: float = 20,          # normalization for # of users in a grid
    mode: str = "marl",             # ["sarl", "marl"]
    **kwargs
) -> np.ndarray:                    # the heatmap of all network entities
    """Return the observation of an agent.

    Args:
        mode=="marl": multi-agent scenario
        mode=="sarl": single-agent scenario

    Returns:
        Flattened observation (user heatmap + relative positions of BSs).
    """
    obs = list()
    n_grids = int(1 + np.floor((2 * bound - 1) / grid_size))
    # hm = np.zeros(shape=(n_grids, n_grids))
    hm = 1e-6 * rng.uniform(size=(n_grids, n_grids))         # random noise
    if mode == "marl":
        keys = ['self', 'uav', 'mbs']
    elif mode == "sarl":
        keys = ['self']

    if locs == {}:
        if mode == "marl":
            return rng.uniform(size=(n_grids * n_grids + 2 * (kwargs['n_uavs'] + kwargs['n_mbss']),))
        elif mode == "sarl":
            return rng.uniform(size=(n_grids * n_grids + 2,))

    for key in keys:
        for i in range(locs[key].shape[-1]):
            obs.append(get_xloc_norm(xloc=locs[key][0, i], bound=bound, grid_size=grid_size))
            obs.append(get_yloc_norm(yloc=locs[key][1, i], bound=bound, grid_size=grid_size))
    xlocs = np.clip(locs['user'][0], -bound + 1, bound - 1)
    ylocs = np.clip(locs['user'][1], -bound + 1, bound - 1)
    for i in range(len(xlocs)):
        col = int(np.floor((xlocs[i] + bound) / grid_size))
        row = int(np.floor((bound - ylocs[i]) / grid_size))
        hm[row, col] += 1
    obs = np.append(obs, hm.flatten() / grid_norm)

    return obs


def get_xloc_norm(xloc, bound: float = 1000, grid_size: float = 100) -> np.ndarray:
    xloc = np.clip(xloc, -bound + 1, bound - 1)
    '''Normalization with point (0,0) in the top-left corner.'''
    return (xloc + bound) / (2 * bound - 1)


def get_yloc_norm(yloc, bound: float = 1000, grid_size: float = 100) -> np.ndarray:
    '''Normalization with point (0,0) in the top-left corner.'''
    yloc = np.clip(yloc, -bound + 1, bound - 1)
    return (bound - yloc) / (2 * bound - 1)
