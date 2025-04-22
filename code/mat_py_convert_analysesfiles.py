import os
import re
import sys
from datetime import datetime

import h5py
import joblib
import numpy as np
import pandas as pd
from matHandler import matHandler

from scripts.mat_py_convert import get_save_dir


def main():
    # rat_name = 'achel180320'
    rat_name = 'hoegaarden181115'
    # conv_ica_weight(rat_name)
    # conv_ica_strength(rat_name)
    # conv_ica_cos_sim(rat_name)
    conv_ica_coact_sig(rat_name)

def conv_ica_coact_sig(rat_name):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=datetime.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name, to_dropbox=True)
    coact_sig_mat = matHandler(analyses_mat_path + '-icaReacCCG_sig.mat')

    mat_generator=coact_sig_mat.getStr(coact_sig_mat.name.icaReacCCG_sig.generator)
    # ica_param_ref = coact_sig_mat.getNum(coact_sig_mat.name.icaReacCCG_sig.param)
    pair_id_ref = coact_sig_mat.getNum(coact_sig_mat.name.icaReacCCG_sig.pairID)
    reg_ref = coact_sig_mat.getNum(coact_sig_mat.name.icaReacCCG_sig.region)

    pair_sig_ref = coact_sig_mat.getNum(coact_sig_mat.name.icaReacCCG_sig.nrem)

    with h5py.File(os.path.expanduser(analyses_mat_path + '-icaReacCCG_sig.mat'), 'r') as f:
        for n in range(pair_sig_ref.shape[0]):

            sig1 = f[pair_sig_ref[n]]['significance'][:]
            sig5 = f[pair_sig_ref[n]]['significance5'][:]
            sig_all = sig5 * 5 - sig1 * 4

            pair_id_all = np.array(f[pair_id_ref[n]][:]-1,dtype=int)
            reg = f[reg_ref[n]][:]
            reg_list = []
            for reg_idx in range(reg.shape[0]):
                reg_list.append(''.join(map(chr,f[reg[reg_idx,0]][:])))

            reg_list = [reg.replace(' cont', '') for reg in reg_list]
            reg_list = [reg.replace('CG', 'Cg') for reg in reg_list]
            reg_list = [reg.replace(' L2/3', 'L23') for reg in reg_list]
            reg_list = [reg.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23') for reg in reg_list]
            reg_list = [reg.replace(' L5', 'L5') for reg in reg_list]
            reg_list = [reg.replace('PrLL', 'PL') for reg in reg_list]
            reg_list = [reg.replace('CgL', 'Cg') for reg in reg_list]
            reg_list = [reg.replace('M2L', 'M2_') for reg in reg_list]
            reg_list = [reg.replace('zonaincerta', 'zona_incerta') for reg in reg_list]
            reg_list = [reg.replace('/', '').replace(' ', '_').replace('\u3000', '_') for reg in reg_list]

            pair_name=np.full_like(pair_id_all,'',dtype=object)
            for nn in range(pair_id_all.shape[0]):
                for mm in range(pair_id_all.shape[1]):
                    pair_name[nn,mm]=reg_list[pair_id_all[nn,mm]]

            across=(pair_name[0, :] != pair_name[1, :])

            significance=sig_all[:,across]
            pair_id=pair_id_all[:,across]
            region=pair_name[:,across]
            template=''.join(map(chr, f[coact_sig_mat.getNum(coact_sig_mat.name.icaReacCCG_sig.template)[n]][:]))
            template = re.sub('([a-z0-9])([A-Z])', r'\1_\2', re.sub('(.)([A-Z][a-z]+)', r'\1_\2', template)).lower()
            template = template.replace('-_', '-')


            joblib_fname = rat_name + '-coact_sig-' + template + '.joblib'

            params = {'templates': template,
                      'state': 'nrem',
                      'generator_matlab': mat_generator[n],
                      'matlab_filename': 'analyses/' + rat_name + '-icaReacCCG_sig.mat'}

            metadata = {'contents': ('metadata', 'significance', 'region','pair_id', 'params'),
                        'generator': __name__,
                        'generate_date': datetime.today().strftime('%Y-%m-%d')}

            joblib.dump((metadata, significance, region, pair_id, params), analyses_dir + joblib_fname, compress=True)

            significance=sig_all[:,~across]
            pair_id=pair_id_all[:,~across]
            region=pair_name[:,~across]

            joblib_fname = rat_name + '-local_coact_sig-' + template + '.joblib'

            params = {'templates': template,
                      'state': 'nrem',
                      'generator_matlab': mat_generator[n],
                      'matlab_filename': 'analyses/' + rat_name + '-icaReacCCG_sig.mat'}

            metadata = {'contents': ('metadata', 'significance', 'region', 'pair_id', 'params'),
                        'generator': __name__,
                        'generate_date': datetime.today().strftime('%Y-%m-%d')}

            joblib.dump((metadata, significance, region, pair_id, params), analyses_dir + joblib_fname, compress=True)


def conv_ica_cos_sim(rat_name):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=datetime.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name, to_dropbox=True)
    ica_cossim = matHandler(analyses_mat_path + '-icaCosSim.mat')
    chance_ref = ica_cossim.getNum(ica_cossim.name.icaCosSim.eachReg.chanceDist)
    region_ref = ica_cossim.getNum(ica_cossim.name.icaCosSim.eachReg.region)
    cossim_ref = ica_cossim.getNum(ica_cossim.name.icaCosSim.eachReg.cosSim)
    template_ref = ica_cossim.getNum(ica_cossim.name.icaCosSim.eachReg.template)

    generator_matlab = ica_cossim.getStr(ica_cossim.name.icaCosSim.generator)

    region_list = []
    chance_dist = []

    val_list = []
    p_list = []
    target_list = []
    template_list = []
    target_id_list = []
    template_id_list = []
    # f=h5py.File(os.path.expanduser(analyses_mat_path + '-icaCosSim.mat'), 'r')
    with h5py.File(os.path.expanduser(analyses_mat_path + '-icaCosSim.mat'), 'r') as f:
        for region_idx in range(region_ref.shape[0]):
            region = ''.join(map(chr, f[region_ref[region_idx]][:]))
            region = region.replace(' cont', '')
            region = region.replace('CG', 'Cg')
            region = region.replace(' L2/3', 'L23')
            region = region.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23')
            region = region.replace(' L5', 'L5')
            region = region.replace('PrLL', 'PL')
            region = region.replace('CgL', 'Cg')
            region = region.replace('M2L', 'M2_')
            region = region.replace('zonaincerta', 'zona_incerta')
            region = region.replace('/', '').replace(' ', '_').replace('\u3000', '_')
            region_list.append(region)

            chance_dist.append(f[chance_ref[region_idx]][:])

            val_ref = f[cossim_ref[region_idx]]['val']
            p_ref = f[cossim_ref[region_idx]]['p']
            ic_name_ref = f[cossim_ref[region_idx]]['icName']
            ic_id_ref = f[cossim_ref[region_idx]]['id']

            template_name = ''.join(map(chr, f[template_ref[region_idx]]['name'][:]))
            template_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                                   re.sub('(.)([A-Z][a-z]+)', r'\1_\2', template_name)).lower()
            template_id = np.array(list(map(int, f[template_ref[region_idx]]['id'][:]))) - 1

            if any(template_id < 0):
                template_id = []

            val_reg = []
            p_reg = []
            target_reg = []
            target_id_reg = []
            template_reg = []
            template_id_reg = []
            for session_idx in range(val_ref.shape[0]):

                target_name = ''.join(map(chr, f[ic_name_ref[session_idx, 0]][:]))
                target_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                                     re.sub('(.)([A-Z][a-z]+)', r'\1_\2', target_name)).lower()
                target_reg.append(target_name.replace('-_', '-'))
                template_reg.append(template_name)

                target_id = np.array(list(map(int, f[ic_id_ref[session_idx, 0]][:]))) - 1

                if any(target_id < 0):
                    val_reg.append([])
                    p_reg.append([])
                    target_id_reg.append([])
                else:
                    val_reg.append(f[val_ref[session_idx, 0]][:])
                    p_reg.append(f[p_ref[session_idx, 0]][:])
                    target_id_reg.append(target_id)

                template_id_reg.append(template_id)

            val_list.append(val_reg)
            p_list.append(p_reg)
            target_list.append(target_reg)
            template_list.append(template_reg)
            target_id_list.append(target_id_reg)
            template_id_list.append(template_id_reg)

    for session_idx in range(len(template_list[0])):
        template_name = template_list[0][session_idx]
        target_name = target_list[0][session_idx]

        cos_sim = []
        p = []
        target_id = []
        template_id = []
        for region_idx in range(len(val_list)):
            cos_sim.append(val_list[region_idx][session_idx])
            p.append(p_list[region_idx][session_idx])
            template_id.append(template_id_list[region_idx][session_idx])
            target_id.append(target_id_list[region_idx][session_idx])

        joblib_fname = rat_name + '-ic_cos_sim-' + target_name + '.joblib'

        params = {'templates': [target_name, template_name],
                  'region': region_list,
                  'ic_id': np.array([target_id, template_id], dtype=object),
                  'generator_matlab': generator_matlab,
                  'matlab_filename': 'analyses/' + rat_name + '-icaCosSim.mat'}

        metadata = {'contents': ('metadata', 'cos_sim', 'p', 'chance_dist', 'params'),
                    'generator': __name__,
                    'generate_date': datetime.today().strftime('%Y-%m-%d')}

        joblib.dump((metadata, cos_sim, p, chance_dist, params), analyses_dir + joblib_fname, compress=True)


def conv_ica_strength(rat_name):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=datetime.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name, to_dropbox=True)
    ica_react_mat = matHandler(analyses_mat_path + '-icaReac.mat')
    strength_ref = ica_react_mat.getNum(ica_react_mat.name.icaReac.strength)
    template_names = ica_react_mat.getStr(ica_react_mat.name.icaReac.tempName)
    template_names = [re.sub('([a-z0-9])([A-Z])', r'\1_\2', re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)).lower() for x in
                      template_names]
    template_names = [x.replace('-_', '-') for x in template_names]

    mat_param = ica_react_mat.getNum(ica_react_mat.name.icaReac.param)
    mat_generator = ica_react_mat.getNum(ica_react_mat.name.icaReac.generator)

    with h5py.File(os.path.expanduser(analyses_mat_path + '-icaReac.mat'), 'r') as f:
        for session_idx in range(strength_ref.shape[0]):
            joblib_fname = rat_name + '-ica_react_strength-' + template_names[session_idx] + '.joblib'

            ica_activation_strength = f[strength_ref[session_idx]][:]
            tbin_size = f[mat_param[session_idx]]['tBinSize'][()][0, 0]
            ncell_min = f[mat_param[session_idx]]['minNcell'][()][0, 0]
            generator_matlab = ''.join(map(chr, f[mat_generator[session_idx]][:]))

            params = {
                'tbin_size': tbin_size,
                'ncell_min': ncell_min,
                'template_name': template_names[session_idx],
                'ica_weight_file': analyses_dir + rat_name + '-ica_weight.joblib',
                'template_index': session_idx,
                'generator_matlab': generator_matlab,
                'matlab_filename': 'analyses/' + rat_name + '-icaReac.mat'}

            metadata = {'contents': ('metadata', 'ica_activation_strength', 'params'),
                        'generator': __name__,
                        'generate_date': datetime.today().strftime('%Y-%m-%d')}

            joblib.dump((metadata, ica_activation_strength, params), analyses_dir + joblib_fname, compress=True)


def conv_ica_weight(rat_name):
    func_name = sys._getframe().f_code.co_name
    print('{time} start {func} with data of {ses}'.format(time=datetime.now().strftime('%H:%M:%S'),
                                                          ses=rat_name, func=func_name))
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name, to_dropbox=True)
    [_, _, clu_info, *_] = joblib.load(base_dir + rat_name + '-ok_spikes.joblib')

    ica_info = matHandler(analyses_mat_path + '-icaReacInfo.mat')

    template = pd.DataFrame()
    template_name = ica_info.getStr(ica_info.name.icaReacInfo.tempName)
    template_name = [re.sub('([a-z0-9])([A-Z])', r'\1_\2', re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)).lower() for x in
                     template_name]
    template_name = [x.replace('-_', '-') for x in template_name]
    template['name'] = template_name

    region_all = ica_info.getNum(ica_info.name.icaReacInfo.region)
    weight_all = ica_info.getNum(ica_info.name.icaReacInfo.weigth)
    time_all = ica_info.getNum(ica_info.name.icaReacInfo.tempTime)

    mat_param = ica_info.getNum(ica_info.name.icaReacInfo.param)
    mat_generator = ica_info.getNum(ica_info.name.icaReacInfo.generator)

    ica_weight = []
    t_start = []
    t_end = []
    with h5py.File(os.path.expanduser(analyses_mat_path + '-icaReacInfo.mat'), 'r') as f:

        for session_idx in range(template['name'].shape[0]):
            temp = f[time_all[session_idx]][()]
            t_start.append(temp[0, 0])
            t_end.append(temp[1, 0])

            temp = f[region_all[session_idx]][()]
            region = []
            for n in range(temp.shape[0]):
                region.append(''.join(map(chr, f[temp[n, 0]][:])))

            region = [reg.replace(' cont', '') for reg in region]
            region = [reg.replace('CG', 'Cg') for reg in region]
            region = [reg.replace(' L2/3', 'L23') for reg in region]
            region = [reg.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23') for reg in region]
            region = [reg.replace(' L5', 'L5') for reg in region]
            region = [reg.replace('PrLL', 'PL') for reg in region]
            region = [reg.replace('CgL', 'Cg') for reg in region]
            region = [reg.replace('M2L', 'M2_') for reg in region]
            region = [reg.replace('zonaincerta', 'zona_incerta') for reg in region]
            region = [reg.replace('/', '').replace(' ', '_').replace('\u3000', '_') for reg in region]

            cell_id = []
            for reg in region:
                cell_id.append(list(map(int, clu_info.index[clu_info.region == reg])))

            temp = f[weight_all[session_idx]][()]
            weight = []
            for n in range(temp.shape[0]):
                weight.append(f[temp[n, 0]][0, :])

            ica_weight.append(pd.DataFrame())
            ica_weight[-1]['weight'] = weight
            ica_weight[-1]['region'] = region
            ica_weight[-1]['cell_id'] = cell_id

        tbin_size = f[mat_param[0]]['tBinSize'][()][0, 0]
        ncell_min = int(f[mat_param[0]]['minNcell'][()][0, 0])
        generator_matlab = ''.join(map(chr, f[mat_generator[0]][:]))

    template['start_t'] = t_start
    template['end_t'] = t_end

    params = {
        'tbin_size': tbin_size,
        'ncell_min': ncell_min,
        'generator_matlab': generator_matlab,
        'matlab_filename': 'analyses/' + rat_name + '-icaReacInfo.mat'}

    metadata = {'contents': ('metadata', 'ica_weight', 'template_info', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, ica_weight, template, params), analyses_dir + rat_name + '-ica_weight.joblib',
                compress=True)


if __name__ == '__main__':
    main()
