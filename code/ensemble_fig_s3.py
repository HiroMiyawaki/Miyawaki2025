# %%
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from my_tools import data_location
import matplotlib.pyplot as plt
from my_tools.pyplot_helper import *
from my_tools.color_preset import *
from datetime import date
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %%


def get_data_s3a():
    # parameters
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()
    swr = {}
    hfo = {}
    c_rip = {}
    region = {}
    for rat in rats:
        data_file = data_dir / rat / "analyses" / \
            f"{rat}-ext_react_modulation_by_oscillation.joblib"
        _, swr[rat], hfo[rat], c_rip[rat], region[rat], t, _ = joblib.load(
            data_file)

    evt_names = ["HFO",  "cRipple", "SWR"]
    evts = [hfo, c_rip, swr]
    region_names = ["BLA",  "PL5", "vCA1",]

    react_mod = {}
    for target_region in region_names:
        for evt, evt_name in zip(evts, evt_names):
            for hc in range(2):
                pooled = np.empty((evt[rat][0].shape[0], 0))
                for rat in rats:
                    idx = region[rat][region[rat][:, 1]
                                      == target_region, 0].astype(int)
                    if evt[rat][hc] is None:
                        continue
                    pooled = np.hstack((pooled, evt[rat][hc][:, idx]))
                react_mod[evt_name, target_region, hc] = pooled

    return react_mod, t, evt_names, region_names

# %%


def plot_s3a(fig, x, y, data, fontsize=8, colors={"HC2": "r", "HC3": "b"}):
    react_mod, t, evt_names, region_names = data

    width = 25
    height = 20
    x_gap = 8
    y_gap = 16
    y_range = {
        ("BLA", "HFO"): (0, 1),
        ("BLA", "cRipple"): (0, 0.6),
        ("BLA", "SWR"): (0, 0.4),
        ("PL5", "HFO"): (0, 0.7),
        ("PL5", "cRipple"): (0, 2),
        ("PL5", "SWR"): (0, 0.5),
        ("vCA1", "HFO"): (0, 0.6),
        ("vCA1", "cRipple"): (0, 0.6),
        ("vCA1", "SWR"): (0, 4)}
    y_tick = {
        ("BLA", "HFO"): [0, 0.5, 1],
        ("BLA", "cRipple"): [0, 0.3, 0.6],
        ("BLA", "SWR"): [0, 0.2, 0.4],
        ("PL5", "HFO"): [0, 0.3, 0.6],
        ("PL5", "cRipple"): [0, 1, 2],
        ("PL5", "SWR"): [0, 0.2, 0.4],
        ("vCA1", "HFO"): [0, 0.3, 0.6],
        ("vCA1", "cRipple"): [0, 0.3, 0.6],
        ("vCA1", "SWR"): [0, 2, 4]}

    hc_names = ["Pre-extinction", "Post-extinction"]
    for reg_idx, target_region in enumerate(region_names):
        for evt_idx, evt_name in enumerate(evt_names):
            ax = subplot_mm([x + evt_idx*(width+x_gap), y +
                            reg_idx*(height+y_gap), width, height], fig=fig)
            ylim = y_range[target_region, evt_name]
            for hc in range(2):
                data = react_mod[evt_name, target_region, hc]
                avg = data.mean(axis=1)
                ste = data.std(axis=1)/np.sqrt(data.shape[1])
                col = colors[f"HC{hc+2}"]

                ax.fill_between(t*1000, avg-ste, avg+ste,
                                alpha=0.5, facecolor=col)
                ax.plot(t*1000, avg, c=col, lw=1)
                if evt_idx == len(evt_names)-1:
                    ax.text(50, ylim[1]-(hc-1)*(ylim[1]-ylim[0])*0.12, f"{hc_names[hc]}",
                            ha="left", va="center", fontsize=fontsize, color=col)
                    ax.plot(-60+np.array([0, 100]),  ylim[1]-(hc-1) *
                            (ylim[1]-ylim[0])*0.12+np.zeros(2), "-", lw=0.5, color=col,
                            clip_on=False)
            ax.set_ylim(ylim)
            ax.set_yticks(y_tick[target_region, evt_name])
            ax.set_xlim(-500, 500)
            ax.set_xlabel(f"Time from {evt_name} peak (ms)")
            if evt_idx == 0:
                ax.set_ylabel(
                    f"{target_region} ensemble\nactivation event rate (1/s)")
            else:
                ax.set_ylabel("")
            box_off(ax)


# %%

def get_data_s3b():
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()
    swr = {}
    hfo = {}
    c_rip = {}
    region = {}
    for rat in rats:
        data_file = data_dir / rat / "analyses" / \
            f"{rat}-ext_react_gain_by_oscillation.joblib"
        _, swr[rat], hfo[rat], c_rip[rat], region[rat],  _ = joblib.load(
            data_file)

    evt_names = ["HFO",  "cRipple", "SWR"]
    evts = [hfo, c_rip, swr]
    region_names = ["BLA",  "PL5", "vCA1",]

    react_gain = {}
    t_val = {}
    p_val = {}
    for target_region in region_names:
        for evt, evt_name in zip(evts, evt_names):
            for hc in range(2):
                pooled = np.empty(0)
                for rat in rats:
                    idx = region[rat][region[rat][:, 1]
                                      == target_region, 0].astype(int)
                    if evt[rat][hc] is None:
                        continue
                    pooled = np.hstack((pooled, evt[rat][hc][idx]))
                react_gain[evt_name, target_region, hc] = pooled
                res = stats.wilcoxon(pooled-1)
                t_val[evt_name, target_region, hc] = res.statistic
                p_val[evt_name, target_region, hc] = res.pvalue

    return react_gain, evt_names, region_names, t_val, p_val
# %%


def plot_s3b(fig, x, y, data, fontsize=8, colors={"HC2": "r", "HC3": "b"}):
    react_gain, evt_names, region_names, t_val, p_val = data
    width = 30
    height = 20
    # x_gap = 8
    y_gap = 16
    legend_width = 15

    for reg_idx, region in enumerate(region_names):
        ax = subplot_mm([x, y+(height+y_gap)*reg_idx, width, height], fig=fig)
        for evt_idx, evt_name in enumerate(evt_names):
            for hc in range(2):
                pooled = react_gain[evt_name, region, hc]
                x_pos = evt_idx+0.2*(2*hc-1)
                col = colors[f"HC{hc+2}"]
                bp = ax.boxplot(
                    pooled, positions=[x_pos], widths=0.35,
                    zorder=5,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={'facecolor': col, 'linewidth': 0},
                    whiskerprops={'color': col},
                    medianprops={'color': "white"},
                    capprops={'color': col})

                p = p_val[evt_name, region, hc]
                if p < 0.001:
                    sig_txt = "***"
                elif p < 0.01:
                    sig_txt = "**"
                elif p < 0.05:
                    sig_txt = "*"
                else:
                    sig_txt = ""

                if sig_txt:
                    y_max = bp["whiskers"][1].get_ydata().max()
                ax.text(x_pos, y_max, sig_txt,
                        color=col, ha="center", va="center", fontsize=fontsize*1.5)
        ax.set_xticks(range(len(evt_names)))
        ax.set_xticklabels(evt_names)
        ax.set_xlim(-0.5, len(evt_names)-0.5)
        ax.plot([-0.5, len(evt_names)-0.5], [1, 1], '-', zorder=0, c="#999999")
        ax.set_ylabel(f"Gain of\n{region} ensemble activation")
        box_off(ax)

        ax = subplot_mm([x+width, y+(height+y_gap) *
                        reg_idx, legend_width, height], fig=fig)
        ax.set_xlim(0, legend_width)
        ax.set_ylim(height, 0)
        hc_names = ["Pre-", "Post-"]
        legend_gaps = 6
        for hc in range(2):
            rect = patches.Rectangle(
                (1, hc*legend_gaps+1.25), 3, 2,
                facecolor=colors[f"HC{hc+2}"],
                edgecolor="none")
            ax.add_patch(rect)
            ax.text(4.5, hc*legend_gaps, f"{hc_names[hc]}",
                    color=colors[f"HC{hc+2}"], fontsize=fontsize,
                    ha="left", va="top")
            ax.text(4.5, hc*legend_gaps+2.5, "extinction",
                    color=colors[f"HC{hc+2}"], fontsize=fontsize,
                    ha="left", va="top")
        ax.axis("off")

        # if p_val[evt_name, target_region, hc] < 0.05:
        #     color = "red"
        # else:
        #     color = "black"
        # ax.text(x,y,f"{evt_name} {target_region} {hc}",color=color)
        # ax.text(x+width/2,y,f"t={t_val[evt_name, target_region, hc]:.2f} p={p_val[evt_name, target_region, hc]:.2f}",color=color)
        # y+=y_gap
# %%


def stats_s3b(ws, data):
    react_gain, evt_names, region_names, t_val, p_val = data

    excel_fonts = excel_font_preset()

    ws.append(["Ensemble", "Oscillatory event",
              "Statistical test", "Statistical value", "P-value"])
    for cell in ws[1]:
        cell.font = excel_fonts["heading"]
    hc_names = ["Pre-extinction", "Post-extinction"]
    for region in region_names:
        for evt_name in evt_names:
            n = len(react_gain[evt_name, region, 0])
            for hc in range(2):
                if p_val[evt_name, region, hc] < 0.001:
                    p_txt = "{:.3e}".format(p_val[evt_name, region, hc])
                else:
                    p_txt = "{:.3f}".format(p_val[evt_name, region, hc])

                ws.append([f"{region} (n = {n})", f"{evt_name}, {hc_names[hc]}", "Wilcoxon signed-rank test",
                           f"T={t_val[evt_name, region, hc]}", p_txt])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = excel_fonts["default"]
    excel_auto_expand(ws)


# %%
data_s3a = get_data_s3a()
data_s3b = get_data_s3b()
# %%
fontsize = 7
panel_fontsize = 9
plt.rcParams['font.size'] = fontsize
fig_s3 = fig_mm([174, 110])
# draw_fig_border()
x = 15
y = 8
homecage_colors = color_preset("homecages")
plot_s3a(fig_s3, x, y, data_s3a, fontsize=fontsize, colors=homecage_colors)
text_mm((x-13, y-5), 'A', fontsize=panel_fontsize, weight='bold', fig=fig_s3)

x = 128
y = 8
plot_s3b(fig_s3, x, y, data_s3b, fontsize=fontsize, colors=homecage_colors)
text_mm((x-14, y-5), 'B', fontsize=panel_fontsize, weight='bold', fig=fig_s3)
plt.show()
fig_dir = Path('~/Dropbox/coact_stability_paper/figures').expanduser()

if not fig_dir.exists():
    fig_dir.mkdir(parents=True)
fig_s3.savefig(fig_dir / f'fig_S3_{date.today().strftime("%Y_%m%d")}.pdf')

# %%
stats_dir = Path('~/Dropbox/coact_stability_paper/stats').expanduser()

wb = Workbook()
ws = wb.active
ws.title = "Figure S3B"
stats_s3b(ws, data_s3b)


wb.save(stats_dir / f'fig_s3_{date.today().strftime("%Y_%m%d")}.xlsx')

# %%
