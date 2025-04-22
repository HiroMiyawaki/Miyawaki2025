# %%
import itertools
import joblib
import numpy as np
from pathlib import Path
from my_tools import data_location
from my_tools.pyplot_helper import *
from my_tools.color_preset import *
from datetime import datetime as dt
import pandas as pd
import scipy
from scipy import stats
from matplotlib import font_manager

import matplotlib.patches as patches
import matplotlib.colors as colors
from datetime import datetime as dt
from scipy import stats
import scikit_posthocs as sp
from my_tools import ART
import itertools
from scipy.signal import butter, filtfilt
from datetime import date
# %%


def get_data_4a():
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()

    # load data
    param = {}
    for idx, rat in enumerate(rats):
        _, temp, param[rat] = joblib.load(
            data_dir / rat / "analyses" / (rat + "-ica_type-cue_and_extinction-ext.joblib"))
        temp.insert(0, 'rat', rat)
        if idx == 0:
            ic_type = temp.copy()
        else:
            ic_type = pd.concat((ic_type, temp))

    # summarize data
    ic_type = ic_type.reset_index(drop=True)
    ic_type['link'] = ic_type['link'].str.capitalize()

    result_post = ic_type.groupby(
        ['region', 'link'])[
        'coact_post'].value_counts().unstack(fill_value=0).stack()

    result_pre = ic_type.groupby(
        ['region', 'link'])[
        'coact_pre'].value_counts().unstack(fill_value=0).stack()

    result = pd.concat(
        [result_pre, result_post],
        axis=1,
        keys=['pre', 'post'])

    regions = ['BLA', 'PL5', 'vCA1']
    links = ['Terminated',  'Transient', 'Maintained', 'Initiated']
    coact_types = ['participated', 'not detected']

    coact_count = {}
    for region in regions:
        temp = result.loc[region]
        cnt = np.zeros((4, 2))
        for link_index, link in enumerate(links):
            for coact_index, coact_type in enumerate(coact_types):
                cnt[link_index, coact_index] = temp.loc[link, coact_type]['post']
        coact_count[region] = cnt

    coact_count["All"] = coact_count["vCA1"] + \
        coact_count["BLA"]+coact_count["PL5"]

    regions.insert(0, "All")
    p_values = {}
    odds_ratios = {}
    for reg_idx, region in enumerate(regions):
        target = np.vstack([coact_count[region][:2, :].sum(
            axis=0), coact_count[region][2:, :].sum(axis=0)])
        odds_ratios[region], p_values[region], *_ = stats.fisher_exact(target)
    fate_types = ["attenuated", "preserved"]

    return coact_count, regions, coact_types, fate_types, p_values, odds_ratios

# %%


def plot_4a(fig, x, y, data, fontsize=8, show_each=False,
            colors={"preserved": "orange", "attenuated": "blue"},
            width=9, height=25, y_margin=16, legend_width=14):

    coact_count, regions, coact_types, fate_types, p_values, odds_ratios = data

    coact_label = {
        'participated': 'Coactivated',
        'not detected': 'Non-coactivated'}

    if show_each:
        regions = [region for region in regions if region != "All"]
    else:
        regions = ["All"]

    for reg_idx, region in enumerate(regions):
        target = np.vstack([coact_count[region][:2, :].sum(
            axis=0), coact_count[region][2:, :].sum(axis=0)])
        p = p_values[region]
        ax = subplot_mm([
            x,
            y+(height+y_margin)*reg_idx,
            width,
            height])

        proportions = target / \
            target.sum(axis=0, keepdims=True)*100
        bottom_pos = np.vstack(([0, 0], proportions)).cumsum(axis=0)

        for is_coact in range(2):
            ax.bar(
                range(coact_count[region].shape[1]),
                proportions[is_coact, :],
                label=fate_types[is_coact],
                bottom=bottom_pos[is_coact, :],
                color=colors[fate_types[is_coact]])
            for is_preserved in range(2):
                if target[is_preserved, is_coact] > 0:
                    ax.text(is_preserved, bottom_pos[is_coact, is_preserved]+proportions[is_coact, is_preserved]/2,
                            int(target[is_coact, is_preserved]), ha="center", va="center",
                            fontsize=fontsize)
        if p < 0.05:
            if p < 0.001:
                sig_str = '***'
            elif p < 0.01:
                sig_str = '**'
            else:
                sig_str = '*'
            ax.plot([0, 0, 1, 1], [102, 104, 104, 102],
                    color='k', linewidth=0.5)
            ax.text(0.5, 110, f"{sig_str}", ha="center",
                    va="top", fontsize=fontsize*1.5)

        if show_each:
            ax.set_title(region, fontsize=fontsize)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([coact_label[x]
                            for x in coact_types], rotation=-30, ha="left")
        box_off()
        ax.set_ylabel("Fraction of ensembles (%)")

        ax = subplot_mm([x+width,
                        y+(height+y_margin)*reg_idx, legend_width, height])
        ax.set_xlim([0, legend_width])
        ax.set_ylim([0, height])
        ax.invert_yaxis()
        legend_rect_width = 4
        legend_rect_height = 2
        legend_rect_margin = 3
        legend_string_height = fontsize * 0.26458333333719

        for idx, category in enumerate(["preserved", "attenuated"]):
            rectangle = patches.Rectangle(
                (2, 1+idx*(legend_rect_height +
                           legend_string_height+legend_rect_margin)),
                legend_rect_width,
                legend_rect_height,
                facecolor=colors[category],
                linestyle='none')
            ax.add_patch(rectangle)
            ax.text(2,
                    1+idx*(legend_rect_height+legend_string_height +
                           legend_rect_margin)+legend_rect_height+1,
                    category.capitalize(),
                    ha="left",
                    va="top",
                    fontsize=fontsize)
            ax.set_axis_off()

        # if reg_idx == len(regions)-1:
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(
        #         handles[::-1], [x.title() for x in labels[::-1]],
        #         bbox_to_anchor=(1, 1), loc='upper left', ncol=1,
        #         frameon=False, fontsize=fontsize,
        #         handlelength=1.25, handleheight=1, labelspacing=0.5, handletextpad=0.2)

# %%


def plot_4a_legend(fig, x, y, fontsize=8, colors={"preserved": "blue",
                                                  "attenuated": "orange"},
                   plot_order=["Preserved", "Attenuated"]):

    cell_height = (fontsize)*0.3528*1.15
    row_label_width = 0.3528*fontsize*5*1.15
    cell_width = 0.3528*fontsize*5*1.15
    column_label_height = (fontsize+1)*0.3528

    font_props = font_manager.FontProperties(family="Sans")

    status = {
        "preserved": [True],
        "attenuated": [False]}
    # status_str = ["\u2716No", "\u2714Yes"]
    status_str = ["No", "Yes"]
    str_color = ["#cc6677", "#33aa88"]

    # column_labels = ["Conditioning"]

    ax = subplot_mm([x, y, row_label_width+cell_width,
                    cell_height*len(plot_order)+3*column_label_height], fig=fig)
    ax.set_xlim(0, row_label_width+cell_width)
    ax.set_ylim(0, cell_height*len(plot_order)+3*column_label_height)

    ax.add_patch(
        patches.Rectangle(
            (row_label_width, 0),
            cell_width*2, column_label_height*3,
            facecolor='white', edgecolor='none'))

    ax.text(row_label_width+cell_width/2, column_label_height*1.5,
            "Detected in\nretention", va='center', ha='center', fontsize=fontsize)

    ax.text(row_label_width/2, column_label_height*1.5,
            "Extinction-\nensemble\nclass", va='center', ha='center', fontsize=fontsize)

    # for n, label in enumerate(column_labels):
    # ax.add_patch(
    #     patches.Rectangle(
    #         (row_label_width+n*cell_width, column_label_height),
    #         cell_width, column_label_height,
    #         facecolor='blue', edgecolor='none'))
    # ax.text(
    #     row_label_width+n*cell_width+cell_width/2,
    #     column_label_height*1.5,
    #     label,
    #     va='center', ha='center',
    #     fontsize=fontsize)

    for idx, label in enumerate(plot_order):
        print(colors)
        ax.add_patch(
            patches.Rectangle(
                (0, idx*cell_height+3*column_label_height),
                row_label_width, cell_height,
                facecolor=colors[label.lower()], edgecolor='none'))
        ax.text(
            0.5*row_label_width,
            (idx+0.5)*cell_height + 3*column_label_height,
            label,
            va='center', ha='center',
            fontsize=fontsize)
        for n, state in enumerate(status[label.lower()]):
            ax.add_patch(
                patches.Rectangle(
                    (row_label_width+cell_width*n, idx *
                     cell_height+3*column_label_height),
                    cell_width, cell_height,
                    facecolor=lighten_color(colors[label.lower()]), edgecolor='none'))
            ax.text(
                row_label_width+cell_width*(n+0.5),
                (idx+0.5)*cell_height+3*column_label_height,
                status_str[state],
                color=str_color[state],
                va='center', ha='center',
                fontproperties=font_props,
                fontsize=fontsize)
    ax.invert_yaxis()
    unity = np.ones(2)
    ax.plot(row_label_width*unity, [0, cell_height*len(plot_order)+3*column_label_height], '-',
            lw=0.5, color="#000000")
    ax.plot((row_label_width+cell_width)*unity,
            [column_label_height, cell_height *
                len(plot_order)+3*column_label_height], '-',
            lw=0.5, color="#000000")
    ax.plot([0, row_label_width+cell_width*2],
            3*column_label_height*unity, '-',
            lw=0.5, color="#000000")
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
# %%


def get_data_4c():
    # parameters
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()

    # load data
    param = {}
    for idx, rat in enumerate(rats):
        _, temp, param[rat] = joblib.load(
            data_dir / rat / "analyses" / (rat + "-ica_type-cue_and_extinction-ext.joblib"))
        temp.insert(0, 'rat', rat)
        if idx == 0:
            ic_type = temp.copy()
        else:
            ic_type = pd.concat((ic_type, temp))

    ic_type = ic_type.reset_index(drop=True)
    ic_type['link'] = ic_type['link'].str.capitalize()

    #  activationでthresholdを超える高さのピークを検出
    peak_id = {}
    peak_height = {}
    peak_time = {}

    for rat in rats:
        _, peak_time[rat], peak_id[rat], peak_height[rat], _ = joblib.load(
            data_dir / rat / "analyses" / (rat + "-ica_react_event-cue_and_extinction-ext.joblib"))

    session_info = {}
    sleep_states = {}
    for rat in rats:
        _, _, session_info[rat], * \
            _ = joblib.load(data_dir / rat / (rat + '-basic_info.joblib'))
        _, sleep_states[rat], * \
            _ = joblib.load(data_dir / rat / (rat + '-sleep_states.joblib'))

    hfo = {}
    swr = {}
    c_rpl = {}
    for rat in rats:
        filename = data_dir / rat / (rat + '-amygdalar_hfo.joblib')
        if filename.exists():
            _, hfo[rat], _ = joblib.load(filename)
        else:
            hfo[rat] = None
        filename = data_dir / rat / (rat + '-hippocampal_swr.joblib')
        if filename.exists():
            _, swr[rat], _ = joblib.load(
                data_dir / rat / (rat + '-hippocampal_swr.joblib'))
        else:
            swr[rat] = None
        filename = data_dir / rat / (rat + '-prefrontal_ripples.joblib')
        if filename.exists():
            _, c_rpl[rat], _ = joblib.load(
                data_dir / rat / (rat + '-prefrontal_ripples.joblib'))
        else:
            c_rpl[rat] = None

    # summarize data
    for rat in rats:
        t_range = session_info[rat][session_info[rat]['name']
                                    == "homecage3"][['start_t', 'end_t']].values[0]
        nrem = sleep_states[rat][sleep_states[rat]['state']
                                 == 'nrem'][['start_t', 'end_t']].values
        nrem = nrem[(nrem[:, 0] >= t_range[0]) & (nrem[:, 1] <= t_range[1]), :]

        idx = np.zeros(0, dtype=int)
        if swr[rat] is not None:
            for n in range(nrem.shape[0]):
                idx = np.hstack((idx, np.where(
                    (swr[rat]["start_t"] > nrem[n, 0]) & (swr[rat]["end_t"] < nrem[n, 1]))[0]))
            swr[rat] = swr[rat].loc[idx][['start_t', 'end_t']].values

        idx = np.zeros(0, dtype=int)
        if hfo[rat] is not None:
            for n in range(nrem.shape[0]):
                idx = np.hstack((idx, np.where(
                    (hfo[rat]["start_t"] > nrem[n, 0]) & (hfo[rat]["end_t"] < nrem[n, 1]))[0]))
            hfo[rat] = hfo[rat].loc[idx][['start_t', 'end_t']].values

        idx = np.zeros(0, dtype=int)
        if c_rpl[rat] is not None:
            for n in range(nrem.shape[0]):
                idx = np.hstack((idx, np.where(
                    (c_rpl[rat]["start_t"] > nrem[n, 0]) & (c_rpl[rat]["end_t"] < nrem[n, 1]))[0]))
            c_rpl[rat] = c_rpl[rat].loc[idx][['start_t', 'end_t']].values

    for type_id in range(3):
        # pooled_cnt = np.zeros(0, dtype=int)
        pooled_rate = np.zeros(0, dtype=float)
        for rat in rats:
            ic_list = ic_type[ic_type["rat"] == rat]["ic"].values
            if type_id == 0:
                evt = swr[rat]
            elif type_id == 1:
                evt = hfo[rat]
            elif type_id == 2:
                evt = c_rpl[rat]
            else:
                continue

            if evt is None:
                # cnt = np.zeros_like(ic_list, dtype=float)
                rate = np.zeros_like(ic_list, dtype=float)
                # cnt[:] = np.nan
                rate[:] = np.nan
            else:
                idx = np.zeros(0, dtype=int)
                for n in range(evt.shape[0]):
                    idx = np.hstack(
                        (idx, np.where((peak_time[rat] > evt[n, 0]) & (peak_time[rat] < evt[n, 1]))[0]))

                cnt = np.zeros(ic_list.max()+1, dtype=int)
                for n in ic_list:
                    cnt[n] = (peak_id[rat][idx] == n).sum()
                rate = cnt / np.diff(evt, axis=1).sum()
            # pooled_cnt = np.hstack((pooled_cnt, cnt))
            pooled_rate = np.hstack((pooled_rate, rate))
        if type_id == 0:
            # ic_type["swr_cnt"] = pooled_cnt
            ic_type["SWRs"] = pooled_rate
        elif type_id == 1:
            # ic_type["hfo_cnt"] = pooled_cnt
            ic_type["HFOs"] = pooled_rate
        elif type_id == 2:
            # ic_type["c_rpl_cnt"] = pooled_cnt
            ic_type["cRipples"] = pooled_rate

    evt_names = ["HFOs", "cRipples", "SWRs",]
    regions = ["All", "BLA", "PL5", "vCA1"]
    categories = ["participated", "not detected"]
    links = ['preserved', 'attenuated']
    react_rate = {}
    anova_results = {}
    posthoc_results = {}
    mann_whitney_results = {}
    for evt_name in evt_names:
        for region in regions:
            if region == "All":
                in_target = ic_type["region"].isin(["PL5", "BLA", "vCA1"])
            else:
                in_target = ic_type["region"] == region

            subset = ic_type[in_target][["link", "coact_post", evt_name]]
            subset.loc[subset["link"].isin(
                ["Maintained", "Initiated"]), "link"] = "preserved"
            subset.loc[subset["link"].isin(
                ["Terminated", "Transient"]), "link"] = "attenuated"
            subset = subset.dropna(how='any', axis=0)

            event_region = f"{evt_name}_{region}"
            mann_whitney_results[event_region] = {}
            patterns = itertools.product(categories, links)
            for pattern_0, pattern_1 in itertools.combinations(patterns, 2):
                val1 = subset[(subset['coact_post'] == pattern_0[0]) & (
                    subset['link'] == pattern_0[1])][evt_name].values
                val2 = subset[(subset['coact_post'] == pattern_1[0]) & (
                    subset['link'] == pattern_1[1])][evt_name].values

                if (len(val1) > 0) & (len(val2) > 0):
                    pattern_combination = frozenset(
                        (f"{x[0]}_{x[1]}" for x in [pattern_0, pattern_1]))
                    mann_whitney_results[event_region][pattern_combination] = scipy.stats.mannwhitneyu(
                        val1, val2, nan_policy='omit')

            for category in categories:
                for link in links:
                    react_rate[f"{region}_{evt_name}_{category}_{link}"] = subset[
                        (subset["link"] == link) &
                        (subset["coact_post"] == category)][evt_name].values

    return evt_names, regions, categories, links, react_rate, mann_whitney_results
# %%


def get_sig_str(p):
    n_str = 0
    if p < 0.05:
        n_str += 1
    if p < 0.01:
        n_str += 1
    if p < 0.001:
        n_str += 1

    return "*"*n_str
# %%


def get_mann_whitney_sig_str(result):
    n_compare = len(result)
    stats_str = {}
    for category in result.keys():
        p = result[category][1]*n_compare
        if p < 0.05:
            stats_str[category] = get_sig_str(p)
    return stats_str


def get_posthoc_sig_str(result):
    target = result[result["p.value"] < 0.05][["contrast", "p.value"]]
    stats_str = {}
    for _, row in target.iterrows():
        contrast = row['contrast']
        p = row['p.value']
        category_pairs = contrast.split(" - ")
        for idx, category in enumerate(category_pairs):
            temp_str = ""
            if 'participated' in category:
                temp_str = "participated_"
            elif 'not detected' in category:
                temp_str = "not detected_"
            else:
                temp_str = "error_"

            if 'preserved' in category:
                temp_str = temp_str + 'preserved'
            elif 'attenuated' in category:
                temp_str = temp_str + 'attenuated'
            else:
                temp_str + 'error'

            category_pairs[idx] = temp_str
    stats_str[frozenset(category_pairs)] = get_sig_str(p)
    return stats_str


def get_anova_sig_str(
        result,
        compare_types={'link': 'preserved', 'coact_post': 'coactivated'}):
    stats_str = {}
    for compare_type in compare_types.keys():
        p = result['Pr(>F)'][compare_type]
        if p < 0.05:
            stats_str[compare_types[compare_type]] = get_sig_str(p)
    return stats_str


def plot_4c(fig, x, y, data, fontsize=8,
            show_each=False,
            colors={"preserved": "plum", "attenuated": "limegreen"},
            width=18, height=25, x_margin=23, y_margin=16, legend_width=15):
    category_names = {
        "participated": "Coactivated",
        "not detected": "Non-coactivated"}

    y_max = {"SWRs_All": 0.95,
             "HFOs_All": 2.0,
             "cRipples_All": 4.5,
             "SWRs_PL5": 0.6,
             "HFOs_PL5": 1.4,
             "cRipples_PL5": 4.1,
             "SWRs_BLA": 0.75,
             "HFOs_BLA": 3.3,
             "cRipples_BLA": 2.7,
             "SWRs_vCA1": 5.5,
             "HFOs_vCA1": 1.3,
             "cRipples_vCA1": 1.3,
             }

    evt_names, regions, categories, links, react_rate, mann_whitney_results = data

    do_posthoc = {}
    stats_str = {}
    for event_region in mann_whitney_results.keys():
        do_posthoc[event_region] = True
        stats_str[event_region] = get_mann_whitney_sig_str(
            mann_whitney_results[event_region])

    if show_each:
        regions = [region for region in regions if region != "All"]
    else:
        regions = ["All"]

    for e_idx, evt_name in enumerate(evt_names):
        for r_idx, region in enumerate(regions):
            plot_left_pos = x+(width+x_margin)*e_idx
            plot_top_pos = y+(height+y_margin)*r_idx
            ax = subplot_mm([plot_left_pos, plot_top_pos, width, height])
            offset = 0.2  # グループ間のオフセット
            upper_whiskers = np.zeros((2, 2))
            x_coordinates = []
            y_coordinates = []
            ensemble_types = []
            for i, category in enumerate(categories):
                for j, link in enumerate(links):
                    values = react_rate[f"{region}_{evt_name}_{category}_{link}"]
                    pos = i + (j-0.5)*2 * offset
                    boxplot = ax.boxplot(
                        values, positions=[pos],
                        widths=0.25, sym='',
                        patch_artist=True)
                    upper_whiskers[i, j] = boxplot['whiskers'][1].get_ydata()[
                        1]
                    boxplot["boxes"][0].set_linestyle('none')
                    boxplot["boxes"][0].set_facecolor(colors[link])
                    boxplot["medians"][0].set_color('w')
                    for whisker in boxplot['whiskers']:
                        whisker.set_color(colors[link])
                    for cap in boxplot['caps']:
                        cap.set_color(colors[link])
                    x_coordinates.append(pos)
                    y_coordinates.append(upper_whiskers[i, j])
                    ensemble_types.append(f"{category}_{link}")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(
                [category_names[x] for x in categories],
                rotation=-30, ha="left")
            ax.set_title(f"Within {evt_name}", fontsize=fontsize)
            if region != "All":
                ax.set_ylabel(f"{region} ensemble \n activation rate (1/s)")
            else:
                ax.set_ylabel(
                    f"Ensemble activation rate (1/s)")

            box_off()
            if f"{evt_name}_{region}" in y_max.keys():
                top_pos = y_max[f"{evt_name}_{region}"]
            else:
                top_pos = ax.get_ylim()[1]
            ax.set_ylim([0, top_pos])

            pairs = []
            sig_txt = []
            for idx in itertools.combinations(range(len(ensemble_types)), 2):
                if frozenset((ensemble_types[idx[0]], ensemble_types[idx[1]])) in stats_str[f"{evt_name}_{region}"].keys():
                    pairs.append(idx)
                    sig_txt.append(stats_str[f"{evt_name}_{region}"][frozenset(
                        (ensemble_types[idx[0]], ensemble_types[idx[1]]))])
            x_range = np.diff(ax.get_xlim())[0]
            add_significance_auto(
                ax, pairs, x_coordinates, y_coordinates,
                significance_text=sig_txt,
                min_vertical_length=top_pos*0.04,
                text_offset=-top_pos*0.09,
                y_offset=top_pos*0.05,
                base_x_offset=x_range * 0.03,
                small_gap=top_pos*0.02,
                line_width=0.5,
                fontsize=fontsize*1.5
            )
            ax = subplot_mm(
                [plot_left_pos+width-3, plot_top_pos, legend_width, height])
            ax.set_xlim([0, legend_width])
            ax.set_ylim([0, height])
            ax.invert_yaxis()
            legend_rect_width = 4
            legend_rect_height = 2
            legend_rect_margin = 3
            legend_string_height = fontsize * 0.26458333333719

            for idx, category in enumerate(["preserved", "attenuated"]):
                rectangle = patches.Rectangle(
                    (2, 1+idx*(
                        legend_rect_height + legend_string_height+legend_rect_margin)),
                    legend_rect_width,
                    legend_rect_height,
                    facecolor=colors[category],
                    linestyle='none')
                ax.add_patch(rectangle)
                ax.text(2,
                        1+idx*(
                            legend_rect_height+legend_string_height +
                            legend_rect_margin)+legend_rect_height+1,
                        category.capitalize(),
                        ha="left",
                        va="top",
                        fontsize=fontsize)
                ax.set_axis_off()
            if "preserved" in stats_str[f"{evt_name}_{region}"].keys():
                ax.plot(
                    [1.5, 0.5, 0.5, 1.5],
                    1+np.array([0, 0, 1, 1])*(
                        legend_rect_height+legend_string_height +
                        legend_rect_margin)+legend_rect_height/2,
                    'k',
                    lw=0.5
                )
                ax.text(
                    1,
                    1+(
                        legend_rect_height+legend_string_height +
                        legend_rect_margin)/2 + legend_rect_height/2,
                    stats_str[f"{evt_name}_{region}"]["preserved"],
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=fontsize*1.5
                )

# %%


def get_data_4b():
    data_dir = data_location.get_rootdir()
    rats = data_location.get_rats()

    rat = rats[7]
    template_idx = 5

    lfp_file = data_dir / rat / 'lfp' / f"{rat}.lfp"
    _, basic_info, session_info, ch_info, lfp_info, *_ =\
        joblib.load(data_dir/rat / f"{rat}-basic_info.joblib")

    ses_t = session_info[session_info["name"] ==
                         "homecage3"][["start_t", "end_t"]].values

    _, react, react_param = \
        joblib.load(data_dir / rat / 'analyses' /
                    (rat + "-ica_react_strength-cue_and_extinction-ext.joblib"))
    _, ica_weight, *_ = joblib.load(data_dir / rat /
                                    'analyses' / (rat + '-ica_weight.joblib'))

    react_t = (
        np.arange(react.shape[0])+0.5)*react_param["tbin_size"]
    react_regions = ica_weight[template_idx]["region"].values

    # _, swr_pandas, _ = joblib.load(data_dir/rat / f"{rat}-hippocampal_swr.joblib")
    # _, hfo_pandas, _ = joblib.load(data_dir/rat / f"{rat}-amygdalar_hfo.joblib")
    # _, crip_pandas, _ = joblib.load(
    #     data_dir/rat / f"{rat}-prefrontal_ripples.joblib")
    # swr = swr_pandas[["start_t", "end_t", "peak_power", "peak_t"]].values
    # hfo = hfo_pandas[["start_t", "end_t", "peak_power", "peak_t"]].values
    # crip = crip_pandas[["start_t", "end_t", "peak_power", "peak_t"]].values

    ch_regions = ch_info["name"].values
    return lfp_file, lfp_info, react, react_t, react_regions, ch_regions

# %%


def plot_4b(
        fig, x, y, data, fontsize=8,
        colors={"BLA": "#ff0000", "vCA1": "#0000ff", "PL5": "#00ff00"},
        times=[46939.6, 40362.84, 46217.4],
        ensembles=[13, 18, 32, 26, 2, 5],
        lfp_ch=[185, 96, 2],
        hfo_band=[90, 180],
        swr_band=[100, 250],
        trace_height=10,
        ensemble_height=7,
        width=36,
        x_gap=1,
        left_margin=9
):

    lfp_file, lfp_info, react, react_t, react_regions, ch_regions = data
    lfp_sf = lfp_info["sample_rate"]

    lowcut, highcut = hfo_band
    order = 5
    nyquist = 0.5 * lfp_sf  # ナイキスト周波数
    low = hfo_band[0] / nyquist
    high = hfo_band[1] / nyquist
    hfo_filt = butter(order, [low, high], btype='band')

    order = 5
    nyquist = 0.5 * lfp_sf  # ナイキスト周波数
    low = swr_band[0] / nyquist
    high = swr_band[1] / nyquist
    swr_filt = butter(order, [low, high], btype='band')

    lfp = np.memmap(
        lfp_file, dtype='int16', mode='r',
        shape=(lfp_info['n_sample'], lfp_info['n_ch']))

    lfp_t = np.arange(lfp_info['n_sample'])/lfp_sf
    uV_per_bit = lfp_info['uV_per_bit']

    ensemble_regions = react_regions[ensembles]
    lfp_regions = ch_regions[lfp_ch]
    ones = np.ones(2)

    for example_index, t_center in enumerate(times):
        t_range = t_center+np.array([-0.25, 0.25])
        lfp_f_range = (lfp_t >= t_range[0]) & (lfp_t < t_range[1])
        subset_lfp = lfp[(lfp_f_range)][:, lfp_ch]*uV_per_bit
        subset_lfp = subset_lfp - np.mean(subset_lfp, axis=0)
        react_f_range = (
            react_t >= t_range[0]-0.02) & (react_t < t_range[1]+0.02)
        subset_react = react[react_f_range, :]
        x_range = t_range-t_range.mean()

        ax = subplot_mm(
            [left_margin+x+(width+x_gap)*example_index, y, width, trace_height])
        ax.axis('off')

        for n in range(3):
            ax.plot(lfp_t[lfp_f_range]-t_range.mean(), subset_lfp[:, n]-1365*n,
                    color=colors[lfp_regions[n]], clip_on=False, lw=0.75)
            if example_index == 0:
                ax.text(x_range[0]-np.diff(x_range)*0.01,
                        -1365*n, lfp_regions[n], ha='right', va='center', fontsize=fontsize,
                        color=colors[lfp_regions[n]])

        ax.set_ylim(-3910, 710)
        y_range = ax.get_ylim()
        ax.set_xlim(x_range)
        ax.plot(0.089*ones, [-3900, -3400], 'k-')
        ax.text(0.095, -3650, "0.5 mV", ha='left',
                va='center', fontsize=fontsize)
        if example_index == 0:
            ax.text(x_range[0]-np.diff(x_range)*0.35,
                    y_range[1]+np.diff(y_range)*0.1,
                    "Wideband LFP",
                    ha='left', va='center',
                    fontsize=fontsize, color='k')

        ones = np.ones(2)
        ax = subplot_mm(
            [left_margin+x+(width+x_gap)*example_index, y+trace_height, width, trace_height])
        for region_index in range(3):
            if lfp_regions[region_index] == "vCA1":
                filtered = filtfilt(
                    *swr_filt, subset_lfp[:, region_index], axis=0)
                band = swr_band
            else:
                filtered = filtfilt(
                    *hfo_filt, subset_lfp[:, region_index], axis=0)
                band = hfo_band
            ax.plot(lfp_t[lfp_f_range]-t_range.mean(), filtered-region_index*195,
                    color=colors[lfp_regions[region_index]], lw=0.75)

            if example_index == 0:
                ax.text(
                    x_range[0]-np.diff(x_range)*0.01,
                    -195*region_index, f"{band[0]}-{band[1]}Hz", ha='right', va='center', fontsize=fontsize,
                    color=colors[lfp_regions[region_index]])
        ax.set_xlim(x_range)
        ax.set_ylim(-660, 200)
        y_range = ax.get_ylim()
        ax.plot(0.089*ones, -650+np.array([0, 100]), 'k-')
        ax.text(0.095, -600, "0.1 mV", ha='left',
                va='center', fontsize=fontsize)
        ax.axis('off')
        if example_index == 0:
            ax.text(x_range[0]-np.diff(x_range)*0.35,
                    y_range[1]+np.diff(y_range)*0.05,
                    "Filtered LFP",
                    ha='left', va='center',
                    fontsize=fontsize, color='k')

        ax = subplot_mm(
            [left_margin+x+(width+x_gap)*example_index, y+trace_height*2, width, ensemble_height])
        reg_count = {}
        for n, ensemble in enumerate(ensembles):
            if ensemble_regions[n] in reg_count.keys():
                reg_count[ensemble_regions[n]] += 1
            else:
                reg_count[ensemble_regions[n]] = 0

            brightness = 0.4 + 0.3*reg_count[ensemble_regions[n]]
            line_col = lighten_color(
                colors[ensemble_regions[n]], brightness=brightness)
            ax.plot(react_t[react_f_range]-t_range.mean(), subset_react[:, ensemble]-3*n,
                    color=line_col, lw=0.75)
        ax.set_xlim(t_range-t_range.mean())
        ax.set_ylim(-21, 30)
        ax.plot([0.1, 0.2], 5*ones, 'k-')
        ax.text(0.15, 5.1, "100ms", ha='center',
                va='bottom', fontsize=fontsize)
        ax.plot(0.089*ones, [5, 15], 'k-')
        ax.text(0.08, 10, "10 z", ha='right', va='center', fontsize=fontsize)
        ax.axis('off')

# %%


def stats_4a(ws, data, show_each=False):
    coact_count, regions, coact_types, fate_types, p_values, odds_ratios = data
    excel_fonts = excel_font_preset()

    if show_each:
        regions = [region for region in regions if region != "All"]
        ws.append(["Region", "Statistical Test",
                  "Statistical value", "P-value"])
    else:
        regions = ["All"]
        ws.append(["Statistical Test", "Statistical value", "P-value"])

    for cell in ws[1]:
        cell.font = excel_fonts["heading"]

    for reg_idx, region in enumerate(regions):
        if p_values[region] < 0.001:
            p_str = f"{p_values[region]:.3e}"
        else:
            p_str = f"{p_values[region]:.3f}"

        target = np.vstack([coact_count[region][:2, :].sum(
            axis=0), coact_count[region][2:, :].sum(axis=0)])
        n_str = ', '.join([str(int(x)) for x in target.sum(axis=0)])

        s_str = coact_count[region]
        if show_each:
            ws.append([
                region,
                f"Fisher's exact test (n = {n_str})",
                f"Odds ratio = {odds_ratios[region]:.3f}",
                p_str])
        else:
            ws.append([
                f"Fisher's exact test (n = {n_str})",
                f"Odds ratio = {odds_ratios[region]:.3f}",
                p_str])

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = excel_fonts["default"]
    excel_auto_expand(ws)


def stats_4c(ws, data, show_each=False):
    category_names = {
        "participated": "Coactivated",
        "not detected": "Non-coactivated"}

    evt_names, regions, categories, links, react_rate, mann_whitney_results = data
    if show_each:
        regions = [region for region in regions if region != "All"]
        ws.append(["Region", "Event", "Ensemble category pair",
                  "Statistical Test", "Statistical value", "P-value without Bonferroni correction"])
    else:
        regions = ["All"]
        ws.append(["Event", "Ensemble category pair",
                  "Statistical Test", "Statistical value", "P-value without Bonferroni correction"])

    excel_fonts = excel_font_preset()
    for cell in ws[1]:
        cell.font = excel_fonts["heading"]

    ensemble_types = []
    ensenble_type_name = []
    for i, category in enumerate(categories):
        for j, link in enumerate(links):
            ensemble_types.append(f"{category}_{link}")
            ensenble_type_name.append(f"[{category_names[category]}, {link}]")

    for region in regions:
        for evt_name in evt_names:
            res = mann_whitney_results[f"{evt_name}_{region}"]

            for idx in itertools.combinations(range(len(ensemble_types)), 2):
                if frozenset((ensemble_types[idx[0]], ensemble_types[idx[1]])) in res.keys():
                    u, p = res[frozenset(
                        (ensemble_types[idx[0]], ensemble_types[idx[1]]))]
                    n_str = ', '.join(
                        [str(len(react_rate[f"{region}_{evt_name}_{ensemble_types[x]}"])) for x in idx])
                    if p < 0.001:
                        p_str = f"{p:.3e}"
                    else:
                        p_str = f"{p:.3f}"

                    if show_each:
                        ws.append([
                            region, evt_name,
                            f"{ensenble_type_name[idx[0]]} - {ensenble_type_name[idx[1]]}",
                            f"Mann-Whitney U test (n = {n_str})",
                            f"U = {u:.3f}",
                            p_str])
                    else:
                        ws.append([
                            evt_name,
                            f"{ensenble_type_name[idx[0]]} - {ensenble_type_name[idx[1]]}",
                            f"Mann-Whitney U test (n = {n_str})",
                            f"U = {u:.3f}",
                            p_str])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = excel_fonts["default"]
    excel_auto_expand(ws)


# %%
data_4a = get_data_4a()
data_4b = get_data_4b()
data_4c = get_data_4c()
# %%
fig_dir = Path('~/Dropbox/coact_stability_paper/figures').expanduser()
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

fontsize = 7
panel_fontsize = 9
plt.rcParams['font.size'] = fontsize


fig_4 = fig_mm((174, 80))
# draw_fig_border()

color_preserved = color_preset("preserved")
x = 17
y = 5

plot_4a_legend(fig_4, x-10, y, fontsize=fontsize, colors=color_preserved)
plot_4a(fig_4, x-4, y+18, data_4a, fontsize=fontsize,
        show_each=False, colors=color_preserved,
        height=42, width=10)
text_mm((x-16, y-2), 'A', fontsize=panel_fontsize, weight='bold', fig=fig_4)

x = 53
y = 3
colors = color_preset("regions")
plot_4b(fig_4, x, y, data_4b, fontsize=fontsize,
        colors=colors)
text_mm((x-11, y), 'B', fontsize=panel_fontsize, weight='bold', fig=fig_4)

x = 60
y = 40
plot_4c(fig_4, x, y, data_4c, fontsize=fontsize,
        show_each=False, colors=color_preserved)
text_mm((x-18, y-2), 'C', fontsize=panel_fontsize, weight='bold', fig=fig_4)


plt.show()
fig_4.savefig(fig_dir / f'fig_4_{date.today().strftime("%Y_%m%d")}.pdf')
# %%
stats_dir = Path('~/Dropbox/coact_stability_paper/stats').expanduser()

wb = Workbook()
ws = wb.active
ws.title = "Figure 4A"
stats_4a(ws, data_4a)
ws = wb.create_sheet("Figure 4C")
stats_4c(ws, data_4c)


if not stats_dir.exists():
    stats_dir.mkdir(parents=True)
wb.save(stats_dir / f'fig_4_{date.today().strftime("%Y_%m%d")}.xlsx')

# %%
fig_dir = Path('~/Dropbox/coact_stability_paper/figures').expanduser()
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

fontsize = 7
panel_fontsize = 9
plt.rcParams['font.size'] = fontsize

fig_S2 = fig_mm((174, 126))
# draw_fig_border()
color_preserved = color_preset("preserved")

x = 15
y = 5
plot_4a(fig_S2, x, y, data_4a, fontsize=fontsize,
        show_each=True, colors=color_preserved)
text_mm((x-14, y-2), 'A', fontsize=panel_fontsize, weight='bold', fig=fig_S2)

x = 56
y = 5
plot_4c(fig_S2, x, y, data_4c, fontsize=fontsize,
        show_each=True, colors=color_preserved, x_margin=26)
text_mm((x-12, y-2), 'B', fontsize=panel_fontsize, weight='bold', fig=fig_S2)

plt.show()
fig_S2.savefig(fig_dir / f'fig_S2_{date.today().strftime("%Y_%m%d")}.pdf')
# %%
stats_dir = Path('~/Dropbox/coact_stability_paper/stats').expanduser()

wb = Workbook()
ws = wb.active
ws.title = "Figure S2A"
stats_4a(ws, data_4a, show_each=True)

ws = wb.create_sheet("Figure S2B")
stats_4c(ws, data_4c, show_each=True)

wb.save(stats_dir / f'fig_s2_{date.today().strftime("%Y_%m%d")}.xlsx')


# %%
