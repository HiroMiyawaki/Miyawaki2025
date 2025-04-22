import numpy as np
import pandas
from scipy import signal, stats


def in_tbin(spikes, t_range, bin_size=0.01, smooth_sigma=0.05, cell_ids=[]):
    """
     get Gaussian smoothed FR in equally distributed time bins

     Parameters
     ----------
     spikes : pandas dataFrame
         must have colummns named "spiketime" and "cluster", that contains timestamps and cluster ids of each spike, respectively.

     t_range : list with 2 elements
         start and end points of FR calculation

     bin_size : double
         size of time bin in sec

     smooth_sigma : double
         sigma of Gaussian kernel for smoothing
         if it is set as zero, no smoothing will be done

     cell_ids : list
         list of cell ids in cluster colummn of "spike"
         when there are cells with no spikes during t_range, it must be specified.

    Returns
     -------
     fr: ndarray
         the (smoothed) FR
     ndarray : t_bin
     ndarray : cell_ids
         time and cell id corresponding to FR matrix
    """

    if len(cell_ids)==0:
        cell_ids = np.unique(spikes['cluster'])

    cbin_edge = np.sort(np.hstack([cell_ids.max() + 0.5, cell_ids]))
    tbin_edge = np.arange(t_range[0], t_range[1], bin_size)

    spk = spikes[
        (spikes["spiketime"] > t_range[0]) & (spikes["spiketime"] < t_range[1])
    ][["spiketime", "cluster"]].to_numpy()
    cnt, *_ = np.histogram2d(spk[:, 0], spk[:, 1], [tbin_edge, cbin_edge])

    if smooth_sigma > 0:
        core = stats.norm.pdf(
            np.arange(-smooth_sigma * 4, smooth_sigma * 4, bin_size), 0, smooth_sigma
        )
        core = np.reshape(core / sum(core), [-1, 1])
        fr = signal.convolve2d(cnt / bin_size, core, mode="same", boundary="symm")
    else:
        fr = cnt / bin_size

    t_bin = (tbin_edge[1:] + tbin_edge[:-1]) / 2

    return fr, t_bin, cell_ids