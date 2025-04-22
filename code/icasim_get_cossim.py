#%%
import collections
import sys
from datetime import datetime as dt

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from my_tools.pyplot_helper import *


# %%
def main():
    # plot_ic_link(do_sort=True)
    rats = get_rat_list()
    for rat in rats:
        get_ic_link(rat, alpha=0.01)
#%%
def plot_nlink():
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func}'.format(time=dt.now().strftime('%H:%M:%S'),
                                       func=func_name))

    [data_dir, fig_dir] = get_dirs()
    output_figname='link-frac'

    rat_list = get_rat_list()
    reg_list = ['PL5', 'vCA1', 'BLA']

    n_link=[[],[],[]]
    n_link_coact=[[],[],[]]
    for rat_name in rat_list:
        analyses_dir = data_dir + rat_name + '/analyses/'
        [_, ic_link, ic_params] = joblib.load(analyses_dir + rat_name + '-ic_link.joblib')


        for [reg_idx, reg] in enumerate(reg_list):
            if not any([x == reg for x in ic_params['region']]):
                continue

            all_ic = set(ic_params['ic_id'][reg][1])
            from_prev = set(ic_link[reg][ic_link[reg][:, 1] == 1, 3])
            to_next = set(ic_link[reg][ic_link[reg][:, 0] == 1, 2])

            coact_next = set(ic_params['ic_id'][reg][1][np.array(ic_params['coact_next_hc'][reg][1]) == 1])

            maintained = from_prev & to_next
            terminated = from_prev - to_next
            initiated = to_next - from_prev
            isolated = all_ic - (from_prev | to_next)

            n_link[reg_idx].append([len(x) for x in [terminated, maintained, initiated, isolated]])
            n_link_coact[reg_idx].append([len(x & coact_next) for x in [terminated, maintained, initiated, isolated]])


    fig = fig_mm()

    plotsize =np.array([40, 40])
    plotgap = np.array([20, 25])
    margin = np.array([20, 10])

    for stat_type in range(3):
        for [reg_idx,reg] in enumerate(reg_list):
            n_total=np.array(n_link[reg_idx]).sum(axis=0)
            n_coact=np.array(n_link_coact[reg_idx]).sum(axis=0)

            if stat_type == 2:
                temp=np.vstack([n_coact,n_total - n_coact])

                dataset = pd.DataFrame(np.vstack([temp[i,:]/sum(temp[i,:]) for i in range(temp.shape[0])]).T*100,
                                       index=['terminated', 'maintained', 'initiated', 'transient'],
                                       columns=['coactivated', 'isolated'])
                y_txt = '% type '
                x_label = ['coactivated', 'isolated']
            elif stat_type == 1:
                dataset = pd.DataFrame([n_coact, n_total - n_coact]/n_total*100,
                                       columns=['terminated', 'maintained', 'initiated', 'transient'],
                                       index=['coactivated', 'isolated'])
                y_txt='% ensemble within each type'
                x_label = ['Terminated', 'Maintained', 'Initiated', 'Transient']
            elif stat_type==0:
                dataset = pd.DataFrame([n_coact,n_total-n_coact],
                                       columns=['terminated', 'maintained', 'initiated', 'transient'],
                                       index=['coactivated','isolated'])
                y_txt='# Ensemble'
                x_label = ['Terminated', 'Maintained', 'Initiated', 'Transient']

            ax = subplot_mm(np.hstack([margin + (plotsize + plotgap) * np.array([reg_idx, stat_type]), plotsize]))

            for i in range(len(dataset)):
                ax.bar(dataset.columns, dataset.iloc[i], bottom=dataset.iloc[:i].sum())
                for j in range(len(dataset.columns)):
                    ax.text(x=j,
                             y=dataset.iloc[:i, j].sum() + (dataset.iloc[i, j] / 2),
                             s='%0.1f'%dataset.iloc[i, j],
                             ha='center',
                             va='bottom'
                            )
            box_off(ax)
            ax.set_xticks(range(len(x_label)))
            ax.set_xticklabels(x_label,rotation=-20,ha='left')
            if reg_idx == len(reg_list)-1 :
                ax.legend(dataset.index,bbox_to_anchor=(1, 1), loc='upper left')
            ax.set(xlabel='Type', ylabel=y_txt)
            ax.set_title(reg)


    text_mm([205,35],'Coactivated:\nExt. ensembles that participated inter-regional CCG\nwith significant peak during the following NREM')
    text_mm([205, 55], 'Isolated:\nExt. ensembles that did not participate inter-regional CCG\nwith significant peak during the following NREM')

    text_mm([215,85],'Terminated:\nExt. ensembles that have link(s)\nwith cond but not with Ret. Ext')
    text_mm([215,100],'Maintained:\nExt. ensembles that have link(s)\nwith both  cond and Ret. Ext')
    text_mm([215,115],'Initiated:\nExt. ensembles that have link(s)\nwith Ret. Ext but not with Cond')
    text_mm([215,130],'Transient:\nExt. ensembles that have no link(s)\n')

    add_generator()

    fig.savefig(fig_dir + '/png/' + output_figname + '.png', dpi=300)

    pdf = PdfPages(fig_dir + output_figname + '.pdf')
    pdf.savefig(fig, dpi=300)
    pdf.close()


#%%

def plot_ic_link(do_sort=True):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func}'.format(time=dt.now().strftime('%H:%M:%S'),
                                       func=func_name))
    rat_list = get_rat_list()
    [data_dir, fig_dir] = get_dirs()
    reg_list = ['PL5', 'vCA1', 'BLA']

    output_figname = 'ic_link'

    fig = fig_mm()
    # plotsize = [25, 38]
    # plotgap = [10, 12]
    # margin = [10, 10]
    plotsize =np.array([25, 38])
    plotgap = np.array([10, 12])
    margin = np.array([10, 10])
    rat_idx = [0, 0, 0]

    for rat_name in rat_list:
        analyses_dir = data_dir + rat_name + '/analyses/'

        [_, ic_link, ic_params] = joblib.load(analyses_dir + rat_name + '-ic_link.joblib')
        for [reg_idx, reg] in enumerate(reg_list):
            if not any([x == reg for x in ic_params['region']]):
                continue

            id_org = ic_params['ic_id'][reg]
            coact = [np.array(x) for x in ic_params['coact_next_hc'][reg]]

            new_link = ic_link[reg].copy()
            new_id = id_org

            cnt = collections.Counter(
                list(ic_link[reg][ic_link[reg][:, 1] == 1, 3]) + list(ic_link[reg][ic_link[reg][:, 0] == 1, 2]))
            cnt = cnt.most_common()

            node = [[], [], []]
            if do_sort:
                nl = [-1, -1, -1]
                for n in range(len(cnt)):
                    c = cnt[n][0]

                    link_id = np.where((new_link[:, 0] == 1) & (new_link[:, 2] == c))
                    new_link[link_id, 2] = nl[1]

                    for pat in link_id[0]:
                        p = new_link[pat, 3];
                        if p < 0:
                            continue
                        pat_id = np.where((new_link[:, 1] == 2) & (new_link[:, 3] == p))
                        new_link[pat_id, 3] = nl[2]
                        node[2].append([nl[2], coact[2][id_org[2] == p][0]])
                        new_id[2][id_org[2] == p] = nl[2]
                        nl[2] -= 1

                    link_id = np.where((new_link[:, 1] == 1) & (new_link[:, 3] == c))
                    new_link[link_id, 3] = nl[1]
                    for pat in link_id[0]:
                        p = new_link[pat, 2];
                        if p < 0:
                            continue
                        pat_id = np.where((new_link[:, 0] == 0) & (new_link[:, 2] == p))
                        new_link[pat_id, 2] = nl[0]
                        node[0].append([nl[0], coact[0][id_org[0] == p][0]])
                        new_id[0][id_org[0] == p] = nl[0]
                        nl[0] -= 1

                    node[1].append([nl[1], coact[1][id_org[1] == c][0]])
                    new_id[1][id_org[1] == c] = nl[1]
                    nl[1] -= 1

                for n in range(3):
                    while any(new_id[n] >= 0):
                        c = max(new_id[n])
                        node[n].append([nl[n], coact[n][id_org[n] == c][0]])
                        new_id[n][id_org[n] == c] = nl[n]
                        nl[n] -= 1
            else:
                for n in range(3):
                    node[n] = [[x, y] for [x, y] in zip(id_org[n], coact[n])]

            col = np.array(['b', 'r'])
            [xx, yy] = divmod(rat_idx[reg_idx], 8)
            # ax = subplot_mm(
            #     [m + (s + g) * p for m, s, g, p in zip(margin, plotsize, plotgap, [yy, reg_idx + xx])] + plotsize)
            ax = subplot_mm(np.hstack([margin + (plotsize + plotgap) * np.array([yy, reg_idx + xx]), plotsize]))

            ax.plot(new_link[:, 0:2].T, new_link[:, 2:4].T, color='k')
            for n in range(3):
                temp = np.array(node[n])
                ax.scatter(n * np.ones([temp.shape[0]]), temp[:, 0], color=col[temp[:, 1]], zorder=3)
            rat_idx[reg_idx] += 1
            ax.set_title('rat' + rat_name[0].upper() + ' ' + reg)
            ax.set(xticks=range(3), yticks=[], xticklabels=['Cond', 'Ext', 'Ret. Ext'])
    text_mm([155, 180], 'Dots indicate cell ensembles identified in each behavioral sessions')
    text_mm([155, 183], 'Red: ensemble that participated inter-regional coactivation in the following NREM', color='r')
    text_mm([155, 186], 'Blue: ensemble that did not participat inter-regional coactivation in the following NREM',
            color='b')
    text_mm([155, 189], 'Cond: conditioning, Ext: cue-extinction, Ret. Ext: retention of extinction', color='k')
    add_generator()

    fig.savefig(fig_dir + '/png/' + output_figname + '.png', dpi=300)

    pdf = PdfPages(fig_dir + output_figname + '.pdf')
    pdf.savefig(fig, dpi=300)
    pdf.close()


# %%
def get_ic_link(rat_name, alpha=0.01):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=dt.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))

    data_dir = '~/Dropbox/analysesPython/data/'
    data_dir = os.path.expanduser(data_dir)
    analyses_dir = data_dir + rat_name + '/analyses/'
    save_filename = analyses_dir + rat_name + '-ic_link.joblib'

    [_, cos_sim, cos_p, _, cossim_param] = joblib.load(analyses_dir + rat_name + '-ic_cossim.joblib')
    following_hc = [1, 2, 3, 3, 3, 3, 4]

    coupled_post = []
    coupled_pre = []
    ses_list = [1, 5, 6]
    template_name = cossim_param['template_name'][ses_list]

    for n in range(3):
        sig_filename = '-coact_sig-' + template_name[n] + '.joblib'
        nrem_ses = following_hc[ses_list[n]]

        [_, sig, _, id, _] = joblib.load(analyses_dir + rat_name + sig_filename)
        sig_post = sig[nrem_ses, :] == 1;
        coupled_post.append(set(id[0, sig_post]) | set(id[1, sig_post]))

        nrem_ses = following_hc[ses_list[n]]-1
        sig_pre = sig[nrem_ses, :] == 1
        coupled_pre.append(set(id[0, sig_pre]) | set(id[1, sig_pre]))

    reg_list = cossim_param['region']

    ic_id = {}
    contributing_post = {}
    contributing_pre = {}
    ic_link = {}
    for reg in reg_list:
        ic_id[reg] = []
        contributing_post[reg] = []
        contributing_pre[reg] = []
        ic_link[reg] = np.zeros([0, 4])
        for n in range(3):
            ic_id[reg].append(np.array(cossim_param['ic_id'][reg][ses_list[n]]))
            contributing_post[reg].append([int(x in coupled_post[n]) for x in cossim_param['ic_id'][reg][ses_list[n]]])
            contributing_pre[reg].append(
                [int(x in coupled_pre[n]) for x in cossim_param['ic_id'][reg][ses_list[n]]])

        for n in range(len(ses_list) - 1):
            temp = np.array(np.where(cos_p[reg][ses_list[n]][ses_list[n + 1]] < alpha))
            if temp.size > 0:
                ses_temp = np.zeros_like(temp)
                ses_temp[0, :] += n
                ses_temp[1, :] += n + 1

                temp[0, :] = ic_id[reg][n][temp[0, :]]
                temp[1, :] = ic_id[reg][n + 1][temp[1, :]]

                ic_link[reg] = np.concatenate([ic_link[reg], np.concatenate([ses_temp, temp]).T], axis=0)

    params = {'template_name': template_name,
              'region': reg_list,
              'ic_id': ic_id,
              'coact_next_hc': contributing_post,
              'coact_prev_hc': contributing_pre,
              'alpha': alpha}

    metadata = {
        'content': ['metadata', 'ic_link', 'params'],
        'generate_date': dt.now().strftime('%Y-%m-%d'),
        'generator': __name__,
        'generator_func': func_name}

    joblib.dump((metadata, ic_link, params), save_filename,
                compress=True)
    # link_sort=[]
    # id_sort=[]
    # id_orig=[[],[],[]]
    # cont_sort=[]
    # for n in range(2):
    #     link_sort.append(np.array([ic_id[n].copy()[link[n][0]], ic_id[n + 1].copy()[link[n][1]]]))
    # for n in range(3):
    #     id_sort.append(ic_id[n].copy())
    #     cont_sort.append(is_cont[n].copy())
    #
    # n=1
    #
    # link_sort[0][1,:]==n
    #
    #
    # col=np.array(['k','r'])
    # for n in range(2):
    #     plt.plot([n,n+1],link_sort[n],color='k')
    # for n in range(3):
    #     plt.scatter(n*np.ones_like(id_sort[n]),id_sort[n],color=col[cont_sort[n]])
    # plt.title(reg)


# %%

def get_cossim(rat_name, n_shuffle=5000):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=datetime.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))

    data_dir = '~/Dropbox/analysesPython/data/'
    data_dir = os.path.expanduser(data_dir)
    analyses_dir = data_dir + rat_name + '/analyses/'
    save_filename = analyses_dir + rat_name + '-ic_cossim.joblib'
    [_, ica_w, temp_info, ica_param] = joblib.load(
        analyses_dir + rat_name + '-ica_weight.joblib')

    reg_set = set()
    for n in range(len(ica_w)):
        reg_set = reg_set | set(ica_w[n].region.values)

    cos_sim = {}
    cos_p = {}
    cos_chance = {}
    ic_id = {}
    for reg in reg_set:
        w = []
        pool_w = []
        ic_id_temp = []
        for n in range(len(ica_w)):
            temp = ica_w[n].weight[ica_w[n].region == reg].values
            ic_id_temp.append(ica_w[n][ica_w[n].region == reg].index.tolist())
            if temp.size > 0:
                w.append(np.stack(temp))
                pool_w.append(np.stack(temp))
            else:
                w.append(np.array([]))

        cos_sim_temp = []
        for n in range(len(w)):
            temp = []
            for m in range(len(w)):
                if (w[n].size > 0) & (w[m].size > 0):
                    temp.append(abs(np.dot(w[n], w[m].T)))
                else:
                    temp.append(np.array([]))
            cos_sim_temp.append(temp)

        all_w = np.vstack(pool_w)
        cos_dist_temp = np.zeros(n_shuffle)
        [n_comp, n_cell] = all_w.shape
        for n in range(n_shuffle):
            idx = np.random.permutation(n_comp)[:2]
            cos_dist_temp[n] = abs(np.dot(all_w[idx[0], np.random.permutation(n_cell)],
                                          all_w[idx[1], np.random.permutation(n_cell)]))

        p_temp = []
        for n in range(len(w)):
            temp = []
            for m in range(len(w)):
                temp.append(
                    np.array(
                        [[(cos_dist_temp > x).mean() for x in cos_sim_temp[n][m][i, :]] for i in
                         range(cos_sim_temp[n][m].shape[0])]
                    )
                )
            p_temp.append(temp)
        cos_sim[reg] = cos_sim_temp
        cos_p[reg] = p_temp
        cos_chance[reg] = np.sort(cos_dist_temp)
        ic_id[reg] = ic_id_temp

    params = {'template_name': temp_info.name.values,
              'region': list(reg_set),
              'ic_id': ic_id,
              'n_shuffle': n_shuffle}

    metadata = {
        'content': ['metadata', 'cos_sim', 'cos_p', 'cos_chance', 'params'],
        'generate_date': datetime.now().strftime('%Y-%m-%d'),
        'generator': __name__,
        'generator_func': func_name}

    joblib.dump((metadata, cos_sim, cos_p, cos_chance, params), save_filename,
                compress=True)


# %%
def get_rat_list(index=[]):
    rat_list = [
        'achel180320',
        'booyah180430',
        'chimay180612',
        'duvel190505',
        'estrella180808',
        'feuillien180830',
        'guiness181002',
        'hoegaarden181115',
        'innis190601',
        'jever190814',
        'karmeliet190901',
        'leffe200124',
        'maredsous200224',
        'nostrum200304',
        'oberon200325'
    ]
    if len(index) > 0:
        rat_list = [rat_list[x] for x in index]

    return rat_list

def get_dirs():
    data_dir = '~/Dropbox/analysesPython/data/'
    data_dir = os.path.expanduser(data_dir)
    fig_dir = '~/Dropbox/analysesPython/fig/'
    fig_dir = os.path.expanduser(fig_dir)
    return data_dir, fig_dir
# %%
if __name__ == '__main__':
    main()
