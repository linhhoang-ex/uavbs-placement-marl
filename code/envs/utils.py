import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, Tuple, List
import os


rng = np.random.default_rng()


def get_horizontal_dist(
    bs_loc: np.ndarray,     # x- and y-coordinates of the BS, shape=(2,)
    users_loc: np.ndarray   # x- and y-coordinates of users, shape=(2 n_users)
) -> np.ndarray:            # horizontal distance to the BS for each user, shape=(n_users,)
    '''Calculate the horizontal distance (in meters) to the base station for each user'''
    delta = bs_loc.reshape(2, 1) - users_loc

    return np.sqrt(np.sum(delta**2, axis=0))


def get_snr_macrobs_db(
    h_dist: np.ndarray,         # horizontal distance (m) to the mBS, shape=(n_users,)
    logF_db: np.float_ = 2,     # std var for the log-normally distributed shadowing
    noise_db: np.float_ = -90,  # total noise power in dBm
    pTx_dBm: np.float_ = 46     # transmit power in dBm of the macro BS
) -> Tuple[Any]:     # snr [dB] for each user, shape=(n_users)
    '''Calculate the path loss [dB] and SNR [dB] for each user via the macro BS's link
    Ref: https://www.arib.or.jp/english/html/overview/doc/STD-T63v9_20/5_Appendix/Rel5/25/25942-530.pdf'''
    p_loss_mean_db = 128.1 + 37.6 * np.log10(h_dist / 1e3)
    p_loss_db = p_loss_mean_db + rng.normal(0, logF_db, len(h_dist))
    snr_mean_db = pTx_dBm - p_loss_mean_db - noise_db
    snr_db = pTx_dBm - p_loss_db - noise_db
    return (snr_db, p_loss_db, snr_mean_db)


def get_snr_uavbs_db(
    h_dist_m: np.ndarray,           # horizontal distance to the drone BS, shape=(n_users,)
    flying_alt: np.float_ = 120,    # flying altitude of the drone BS
    ref_pw_db: np.float_ = -47,     # theta: reference signal power [dB] at d0 = 1 m
    noise_db: np.float_ = -90,      # total noise power in dBm
    pTx_dBm: np.float_ = 30,        # transmit power in dBm of the drone BS
    kappa: np.float_ = 50,          # coefficient for the Rician channel effect
    p_loss_coeff: np.float_ = 2.7   # path loss's coefficient
) -> Tuple[Any]:
    '''Calcualte the path loss [dB] and SNR [dB] for each user via the drone BS's link

    References:
    - https://doi.org/10.1109/TWC.2019.2926279 (for the comm. model)
    - https://doi.org/10.1109/LWC.2017.2710045 (experimental values for the path loss coefficient)
    '''
    dist_m = np.sqrt(h_dist_m**2 + flying_alt**2)
    p_loss_db = 10 * p_loss_coeff * np.log10(dist_m)
    psi_rician = np.sqrt(kappa / (1 + kappa)) \
        + np.sqrt(1 / (1 + kappa)) * rng.normal(size=len(h_dist_m))
    snr_db = pTx_dBm + ref_pw_db + to_dB(psi_rician**2) - p_loss_db - noise_db
    snr_mean_db = pTx_dBm + ref_pw_db - p_loss_db - noise_db

    return (snr_db, p_loss_db, snr_mean_db)


def get_drate_bps(
    snr_db: np.ndarray,         # SNR [dB] of all users, shape=(n_users,)
    bw_mhz: np.ndarray = 20,    # bandwidth in MHz for each user, shape=(n_users,)
) -> np.ndarray:                # data rate [bps] for each user, shape = n_users
    '''Calculate the data rate [bps] via the SNR [dB] and channel bandwidth [MHz]'''
    return bw_mhz * 1e6 * np.log2(1 + convert_from_dB(snr_db))


def to_dB(val):
    '''Convert real values to dB'''
    return 10 * np.log10(val)


def convert_from_dB(val_dB):
    '''Convert dB to real values'''
    return 10 ** (val_dB / 10)


def save_img_from_rgba_arr(
    rgba_arr: np.ndarray, fpath: str = 'tmp/tmp.png',
    figsize: Tuple = (3, 3), dpi: int = 100, transparent=True
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


def create_cmap_alpha(cmap_name):
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
    figsize: Tuple = (3, 3),        # image size in inches
    dpi: float = 100,               # dots per inch (dpi), output image ~ figsize * dpi
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
    fig = plt.figure(figsize=figsize)
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


def gen_user_locs(
    hotspot_dict: List[Tuple[float, float, float, float]],
    bound: float = 1000
) -> np.ndarray:
    '''Generate user locations around some hot spots, each hot spot is represented
    as a tuple of (x0, y0, stddev, nusers). Output shape = (2, n_users)'''
    xlocs = np.array([])
    ylocs = np.array([])
    for x0, y0, stddev, nusers in hotspot_dict:
        xlocs = np.concatenate((xlocs, rng.normal(x0, stddev, nusers)))
        ylocs = np.concatenate((ylocs, rng.normal(y0, stddev, nusers)))

    for loc in [xlocs, ylocs]:
        loc = np.clip(loc, -bound, bound)

    return np.asarray([xlocs, ylocs])


def user_association_greedy_drate(
    locs: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Associate users to macro and drone BSs based on each link's data rate.
    Input: a dictionary like {'user': np.ndarray, 'mbs': np.ndarray, 'uav': np.ndarray}'''
    user_locs = locs['user']
    bs_locs = np.concatenate((locs['mbs'], locs['uav']), axis=-1)
    n_users = user_locs.shape[-1]
    n_mbss = locs['mbs'].shape[-1]
    n_uavs = locs['uav'].shape[-1]

    # Calculate the data rate on each links
    drates = np.zeros(shape=(n_mbss + n_uavs, n_users))
    for i in range(n_mbss + n_uavs):
        h_dist_ = get_horizontal_dist(bs_locs[:, i], user_locs)
        snr_ = get_snr_macrobs_db(h_dist_)[0] if i < n_mbss else get_snr_uavbs_db(h_dist_)[0]
        drates[i, :] = get_drate_bps(snr_)

    # Assign users to base stations based on the drate
    drates_eff = np.max(drates, axis=0)
    indexes = np.argmax(drates, axis=0)

    return drates_eff, indexes, drates
