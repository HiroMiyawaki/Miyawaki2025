# %%
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from my_tools import data_location
import datetime
# %%
# parameters
data_dir = data_location.get_rootdir()
rats = data_location.get_rats()

ensemble_id = {}
ensemble_react_time = {}
region = {}
for rat in rats:
    _, ensemble_react_time[rat], ensemble_id[rat], _, _ = joblib.load(
        data_dir / rat / "analyses" / (rat + "-ica_react_event-cue_and_extinction-ext.joblib"))
    _, ic_type, _ = joblib.load(
        data_dir / rat / "analyses" / (rat + "-ica_type-cue_and_extinction-ext.joblib"))
    region[rat] = ic_type[["ic", "region"]]

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

# %%
#
# summarize data
swr_sub = [{}, {}]
hfo_sub = [{}, {}]
c_rpl_sub = [{}, {}]
nrem_sub = [{}, {}]
ensemble_react_time_sub = [{}, {}]
ensemble_id_sub = [{}, {}]
for evt_type in range(3):
    if evt_type == 0:
        evt = swr
        evt_sub = swr_sub
    elif evt_type == 1:
        evt = hfo
        evt_sub = hfo_sub
    else:
        evt = c_rpl
        evt_sub = c_rpl_sub

    for hc in range(2):
        if hc == 0:
            target_ses = "homecage2"
        else:
            target_ses = "homecage3"

        for rat in rats:
            t_range = session_info[rat][session_info[rat]['name']
                                        == target_ses][['start_t', 'end_t']].values[0]
            nrem = sleep_states[rat][sleep_states[rat]['state']
                                     == 'nrem'][['start_t', 'end_t']].values
            nrem = nrem[(nrem[:, 0] >= t_range[0]) &
                        (nrem[:, 1] <= t_range[1]), :]
            
            if nrem[0, 0] < t_range[0]:
                nrem[0,0] = t_range[0]
            if nrem[-1, 1] > t_range[1]:
                nrem[-1,1] = t_range[1]
            nrem_sub[hc][rat] = nrem
            ensemble_react_time_sub[hc][rat] = ensemble_react_time[rat][(
                ensemble_react_time[rat] >= t_range[0]) & (ensemble_react_time[rat] <= t_range[1])]
            ensemble_id_sub[hc][rat] = ensemble_id[rat][(
                ensemble_react_time[rat] >= t_range[0]) & (ensemble_react_time[rat] <= t_range[1])]

            idx = np.zeros(0, dtype=int)
            if evt[rat] is not None:
                for n in range(nrem.shape[0]):
                    idx = np.hstack((idx, np.where(
                        (evt[rat]["start_t"] > nrem[n, 0]) & (evt[rat]["end_t"] < nrem[n, 1]))[0]))
                evt_sub[hc][rat] = evt[rat].loc[idx][['start_t','end_t','peak_t']].values
            else:
                evt_sub[hc][rat] = None

# %%
bin_size = 0.02
n_half = 26
bin_t = np.arange(-(n_half+0.5)*bin_size, (n_half+1.5)*bin_size, bin_size)

peth_swr = [{}, {}]
peth_hfo = [{}, {}]
peth_c_rpl = [{}, {}]
min_t = bin_t.min()
max_t = bin_t.max()
for rat in rats:
    bin_ic = np.sort(np.unique(ensemble_id[rat]))-0.5
    bin_ic = np.append(bin_ic, bin_ic[-1]+1)
    for evt_type in range(3):
        if evt_type == 0:
            evt_sub = swr_sub
            peth = peth_swr
        elif evt_type == 1:
            evt_sub = hfo_sub
            peth = peth_hfo
        else:
            evt_sub = c_rpl_sub
            peth = peth_c_rpl

        for hc in range(2):

            if evt_sub[hc][rat] is None:
                peth[hc][rat] = None
                continue

            cnt = np.zeros((len(bin_t)-1, len(bin_ic)-1))
            peaks = evt_sub[hc][rat][:,2]
            react_t=ensemble_react_time_sub[hc][rat]
            react_id=ensemble_id_sub[hc][rat]
            for t in peaks:
                interest_t = react_t[(react_t >= t+min_t) & (react_t <= t+max_t)]-t
                interest_id = react_id[(react_t >= t+min_t) & (react_t <= t+max_t)]
                cnt_each, * \
                    _ = np.histogram2d(
                        interest_t,interest_id, bins=[bin_t, bin_ic])
                cnt = cnt+cnt_each
            peth[hc][rat] = cnt/len(evt_sub[hc][rat])/bin_size

# %%
gain_swr = [{},{}]
gain_hfo = [{},{}]
gain_c_rpl = [{},{}]
# rate_swr = [{},{}]
# rate_hfo = [{},{}]
# rate_c_rpl = [{},{}]
# rate_nrem = [{},{}]
for rat in rats:
    bin_ic = np.sort(np.unique(ensemble_id[rat]))-0.5
    bin_ic = np.append(bin_ic, bin_ic[-1]+1)
    for hc in range(2):
        nrem=nrem_sub[hc][rat]
        react_t = ensemble_react_time_sub[hc][rat]
        react_id = ensemble_id_sub[hc][rat]

        cnt_each, *_ = np.histogram2d(react_t, react_id, bins=[nrem.reshape(-1), bin_ic])
        nrem_cnt = cnt_each[::2,:].sum(axis=0)
        nrem_dur = np.diff(nrem, axis=1).sum()
        # rate_nrem[hc][rat] = nrem_cnt/nrem_dur
        for evt_type in range(3):
            if evt_type == 0:
                evt_sub = swr_sub
                gain = gain_swr
                # rate = rate_swr
            elif evt_type == 1:
                evt_sub = hfo_sub
                gain = gain_hfo
                # rate = rate_hfo
            else:
                evt_sub = c_rpl_sub
                gain = gain_c_rpl
                # rate = rate_c_rpl

            if evt_sub[hc][rat] is None:
                gain[hc][rat] = None
                # rate[hc][rat] = None
                continue
            cnt_each, *_ = np.histogram2d(react_t, react_id, bins=[evt_sub[hc][rat][:, :2].reshape(-1), bin_ic])

            evt_cnt = cnt_each[::2, :].sum(axis=0)
            evt_dur = np.diff(evt_sub[hc][rat][:, :2], axis=1).sum()
            # rate[hc][rat] = evt_cnt/evt_dur
            gain[hc][rat] = (evt_cnt/evt_dur)/(nrem_cnt/nrem_dur)

# %%
t = (bin_t[:-1]+bin_t[1:])/2
for rat in rats:
    save_file = data_dir / rat / "analyses" / \
        f"{rat}-ext_react_modulation_by_oscillation.joblib"

    params = {"bin_size": bin_size,
              "target_session": ["homecage2", "homecage3"]}

    metadata = {
        "contents": ["metadata", "swr_mod", "hfo_mod", "c_rip_mod", "region", "t", "params"],
        "generator": "react_modulation_by_oscillation.py",
        "gnerate_date": datetime.datetime.now().strftime('%Y-%m-%d')
    }
    swr_mod = []
    hfo_mod = []
    c_rip_mod = []
    for hc in range(2):
        swr_mod.append(peth_swr[hc][rat])
        hfo_mod.append(peth_hfo[hc][rat])
        c_rip_mod.append(peth_c_rpl[hc][rat])
    reg = region[rat].values

    joblib.dump((metadata, swr_mod, hfo_mod, c_rip_mod, reg, t,params),
                save_file, compress=True)

# %%
for rat in rats:
    save_file = data_dir / rat / "analyses" / \
        f"{rat}-ext_react_gain_by_oscillation.joblib"

    swr_gain = []
    hfo_gain = []
    c_rip_gain = []
    for hc in range(2):
        swr_gain.append(gain_swr[hc][rat])
        hfo_gain.append(gain_hfo[hc][rat])
        c_rip_gain.append(gain_c_rpl[hc][rat])
    reg = region[rat].values
    metadata = {
        "contents": ["metadata", "swr_gain", "hfo_gain", "c_rip_gain", "region", "params"],
        "generator": "react_modulation_by_oscillation.py",
        "gnerate_date": datetime.datetime.now().strftime('%Y-%m-%d')}

    params = {"target_session": ["homecage2", "homecage3"]}

    joblib.dump((metadata, swr_gain, hfo_gain, c_rip_gain, reg, params),
                save_file, compress=True)
