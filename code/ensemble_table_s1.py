# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_tools.pyplot_helper import *
from my_tools import data_location
# %%
data_dir = data_location.get_rootdir()
rats = data_location.get_rats()


# %%
def get_data_s1():
    ses_names = ["conditioning", "cue_and_extinction-ext",
                 "retention_of_extinction"]
    ensemble_names = {
        "conditioning": "Conditioning",
        "cue_and_extinction-ext": "Extinction",
        "retention_of_extinction": "Retention"}
    ensemble_regions = [[], [], []]

    for rat in rats:

        _, ica_weight, template_info, _ = joblib.load(
            data_dir / rat / "analyses" / f"{rat}-ica_weight.joblib")

        for n, ses_name in enumerate(ses_names):
            idx = np.where(template_info["name"] == ses_name)[0][0]
            ensemble_regions[n] = ensemble_regions[n] + \
                list(ica_weight[idx]["region"].values)

    return ensemble_regions, ensemble_names, ses_names

# %%


def plot_table_s1(fig, data, fontsize=8):
    cell_width = 30
    header_width = 15
    cell_height = 6
    margin = 2
    width = header_width + cell_width*3+margin*2
    height = cell_height*4+margin*2

    ensemble_regions, ensemble_names, ses_names = data

    ax = subplot_mm((0, 0, width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    zeros = np.zeros(2)
    for temp_id, temp_name in enumerate(ses_names):
        ax.text(header_width+cell_width*(temp_id+0.5)+margin, cell_height*0.5+margin,
                f"{ensemble_names[temp_name]}-ensembles", ha="center", va="center",
                fontsize=fontsize)

    for reg_idx, region in enumerate(regions):
        ax.text(header_width*0.5+margin, cell_height*(reg_idx+1.5)+margin,
                region, ha="center", va="center",
                fontsize=fontsize)
        for temp_id, temp_name in enumerate(ses_names):
            n_ensemble = sum([x == region for x in ensemble_regions[temp_id]])
            ax.text(header_width+cell_width*(temp_id+0.5)+margin, cell_height*(reg_idx+1.5)+margin,
                    f"{n_ensemble} ensembles", ha="center", va="center",
                    fontsize=fontsize)
    ax.plot([0, width], margin+zeros, lw=1, color="black", clip_on=False)
    ax.plot([0, width], cell_height*4+margin+zeros,
            lw=1, color="black", clip_on=False)
    ax.plot([header_width+margin, width-margin], cell_height+margin+zeros,
            lw=0.5, color="black", clip_on=False)

    ax.axis("off")


# %%
fontszie = 7
table_s1 = fig_mm([width, height])
data_s1 = get_data_s1()
plot_table_s1(table_s1, data_s1, fontsize=fontszie)

fig_dir = Path('~/Dropbox/coact_stability_paper/figures').expanduser()
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

table_s1.savefig(fig_dir / f'table_S1_{date.today().strftime("%Y_%m%d")}.pdf')

# %%
