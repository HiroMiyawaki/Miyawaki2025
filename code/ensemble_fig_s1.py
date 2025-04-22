# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
import numpy as np
from pathlib import Path
from my_tools import data_location
from my_tools import get_fr
from my_tools.pyplot_helper import *
import matplotlib.patches as patches
from datetime import datetime as dt
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as patches
# LogNormをimport
import math
from my_tools import color_preset
import matplotlib.patches as patches
from datetime import date
from openpyxl import Workbook
from openpyxl.styles import NamedStyle, Font

# %%


def get_data_s1():
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()
    rat = rats[12]
    region = 'PL5'
    n_shuffle = 10000
    dims = [2, 3, 5, 10, 30, 50, 100]

    _, cos_sim, cos_p, cos_chance, params = joblib.load(
        data_dir / rat / 'analyses' / (rat + '-ic_cossim.joblib'))
    _, ica_weight, *_ = joblib.load(
        data_dir / rat / 'analyses' / (rat + '-ica_weight.joblib'))

    temp = []
    for n in range(len(ica_weight)):
        temp.append(np.vstack(ica_weight[n].query(
            'region==@region')['weight'].to_numpy()))

    weight_pool = np.vstack(temp)

    c_cnt = {}
    c_edge = np.linspace(0, 1, 201, endpoint=True)
    c_bin = (c_edge[:-1]+c_edge[1:])/2
    np.random.seed(1)
    for d in dims:
        x = 2*np.random.rand(n_shuffle, d)-1
        y = 2*np.random.rand(n_shuffle, d)-1

        c = (x * y).sum(axis=1)/(x**2).sum(axis=1)**0.5/(y**2).sum(axis=1)**0.5

        c_cnt[d], _ = np.histogram(np.abs(c), bins=c_edge)

    def cs(a, b): return np.abs(np.dot(a, b) /
                                (np.linalg.norm(a) * np.linalg.norm(b)))

    n_weight = weight_pool.shape[0]
    n_cell = weight_pool.shape[1]
    rng = np.random.default_rng()

    null_cs = np.zeros(n_shuffle)

    for x in range(n_shuffle):
        idx = rng.choice(n_weight, size=2, replace=False)
        null_cs[x] = cs(rng.permutation(weight_pool[idx[0], :]),
                        rng.permutation(weight_pool[idx[1], :]))

    null_cs = np.sort(null_cs)[::-1]
    null_dist, _ = np.histogram(null_cs, bins=c_edge)

    null_th = null_cs[math.floor(n_shuffle*0.01)]

    real_cnt = {}

    real_cnt[1, 5], _ = np.histogram(
        cos_sim[region][1][5].reshape(-1), bins=c_edge)
    real_cnt[5, 6], _ = np.histogram(
        cos_sim[region][5][6].reshape(-1), bins=c_edge)

    return c_cnt, c_bin, null_dist, null_th, real_cnt, n_shuffle, n_cell


def plot_s1(fig, x, y, data, fontsize=8):
    width = 50
    height = 50
    gap_x = 35

    c_cnt, c_bin, null_dist, null_th, real_cnt, n_shuffle, n_cell = data

    ax = subplot_mm([x, y, width, height], fig=fig)
    cmap = cm.get_cmap('viridis', len(c_cnt))
    for n, (k, v) in enumerate(c_cnt.items()):
        ax.plot(c_bin, np.cumsum(v/n_shuffle*100), label=k, color=cmap(n))
    ld = ax.legend(ncol=3, bbox_to_anchor=[
        0.55, 0.05], loc='lower left', borderaxespad=0, fontsize=fontsize)
    ld.set_title('Dimension')
    ld.get_title().set_fontsize(fontsize)
    ax.set_xticks(np.arange(0, 1.5, 0.5))

    ax.set_ylabel('Cumulative fraction\nof vector pairs (%)')
    ax.set_xlabel('Absolute cosine similarity')
    box_off()
    ax.set_title('Random vectors')
    ax = subplot_mm([x+width+gap_x, y, width, height], fig=fig)

    ax.plot(null_th*np.ones(2), [0, 99.5], color='#000000', label='')
    ax.plot(c_bin, np.cumsum(null_dist/n_shuffle*100),
            label=f'shuffle (n = {n_cell} cells)', color=cmap(5))
    ax.plot(c_bin, np.cumsum(
        real_cnt[1, 5]/real_cnt[1, 5].sum()*100), label='Conditioning vs Extinction',
        color="#ec932b")
    ax.plot(c_bin, np.cumsum(
        real_cnt[5, 6]/real_cnt[5, 6].sum()*100), label='Extinction vs Retention',
        color="#f02e97")
    ax.text(null_th+0.025, 80, '99th percentile', fontsize=fontsize)

    ld = ax.legend(ncol=1, bbox_to_anchor=[
        0.5, 0.05], loc='lower left', borderaxespad=0, fontsize=fontsize)

    ax.set_xticks(np.arange(0, 1.5, 0.5))

    ax.set_ylabel('Cumulative fraction\nof ensemble pairs (%)')
    ax.set_xlabel('Absolute cosine similarity')
    ax.set_title('Actual data')
    box_off()


    # text_mm(
    #     [30, 140], r'$Cosine \: similarity = \frac{x \cdot y}{|x||y|}$', size=18)
# %%
data_s1 = get_data_s1()
# %%
fontsize = 7
panel_fontsize = 9
plt.rcParams['font.size'] = fontsize
fig_S1 = fig_mm([174, 65])

x = 15
y = 5
plot_s1(fig_S1, x, y, data_s1, fontsize=fontsize)
plt.show()

fig_dir = Path('~/Dropbox/coact_stability_paper/figures').expanduser()
# stats_dir = Path('~/Dropbox/coact_stability_paper/stats').expanduser()

fig_S1.savefig(fig_dir / f'fig_S1_{date.today().strftime("%Y_%m%d")}.pdf')
