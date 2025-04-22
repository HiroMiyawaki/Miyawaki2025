import os
import re
from datetime import datetime

import h5py
import joblib
import numpy as np
import pandas as pd
from matHandler import matHandler


# %%
# test dataset
def main():
    # rat_name = 'achel180320'
    rat_name = 'hoegaarden181115'
    conv_all(rat_name)


# %%
def conv_all(rat_name):
    print('{time} start converting data of {data}'.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                          data=rat_name))
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    if not os.path.isdir(analyses_dir):
        print('\tmade data directory')
        os.makedirs(analyses_dir)

    funcs = [
        conv_basic_info,
        conv_ok_spikes,
        conv_ng_spikes,
        conv_sleep_states,
        conv_cues,
        conv_shocks,
        conv_videoframe,
        conv_clock,
        conv_heartbeat,
        conv_freeze,
        conv_hfo,
        conv_swr,
        conv_pfc_ripples,
        conv_pfc_fast_gamma,
        conv_pfc_slow_gamma,
        conv_pfc_slow_wave,
        conv_pfc_k_complex,
        conv_pfc_spindle,
        conv_pfc_off
    ]
    for func in funcs:
        print('\t{time} start {func}'.format(time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), func=func.__name__))
        func(rat_name)


# %%
def save_dir(rat_name,to_dropbox=False):
    basic_mat = matHandler('~/data/Fear/triple/' + rat_name + '/' + rat_name + '.basicMetaData.mat')
    base_mat_path_set = basic_mat.getStr(basic_mat.name.basicMetaData.Basename)
    analyses_mat_path_set = basic_mat.getStr(basic_mat.name.basicMetaData.AnalysesName)

    if to_dropbox:
        data_root_set = os.path.expanduser('~/Dropbox/analysesPython/data/')
    else:
        data_root_set = '/Volumes/Data/'

    base_dir_set = data_root_set + rat_name + '/'
    analyses_dir_set = data_root_set + rat_name + '/analyses/'

    return base_mat_path_set, analyses_mat_path_set, base_dir_set, analyses_dir_set, data_root_set


# %%
def conv_basic_info(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = [rat_name + '.basicMetaData.mat',
                       rat_name + '.AnimalMetadata.mat']

    basic_info = {}
    ch_info = pd.DataFrame()
    lfp_info = {}
    dat_info = {}
    video_info = {}

    animal_mat = matHandler(base_mat_path + '.AnimalMetadata.mat')
    basic_mat = matHandler(base_mat_path + '.basicMetaData.mat')
    basic_info['wavelet_ch'] = (int(basic_mat.getNum(basic_mat.name.basicMetaData.Ch.amyGamma)) - 1,
                                int(basic_mat.getNum(basic_mat.name.basicMetaData.Ch.hpcTheta)) - 1,
                                int(basic_mat.getNum(basic_mat.name.basicMetaData.Ch.pfcDelta)) - 1)
    basic_info['wavelet_ch_name'] = ('amygdala_gamma', 'hippocampus_theta', 'prefrontal_delta')

    # basic_info['t_range_ttl']=basic_mat.getNum(basic_mat.name.basicMetaData.detectionintervals.ttl)
    basic_info['t_range'] = tuple(basic_mat.getNum(basic_mat.name.basicMetaData.detectionintervals.lfp))
    basic_info['base_dir'] = base_dir
    basic_info['analyses_dir'] = analyses_dir
    # basic_info['base_dir'] = '~/Dropbox/analysesPython/data/' + rat_name + '/'
    # basic_info['analyses_dir'] = '~/Dropbox/analysesPython/data/' + rat_name + '/analyses/'
    basic_info['rat_name'] = rat_name

    lfp_info['sample_rate'] = basic_mat.getNum(basic_mat.name.basicMetaData.SampleRates.lfp)
    lfp_info['n_sample'] = int(basic_mat.getNum(basic_mat.name.basicMetaData.nSample.lfp))
    # lfp_info['filepath'] = basic_mat.getStr(basic_mat.name.basicMetaData.lfp)
    lfp_info['filepath'] = 'lfp/' + rat_name + '.lfp'
    lfp_info['n_ch'] = int(basic_mat.getNum(basic_mat.name.basicMetaData.nCh))
    lfp_info['uV_per_bit'] = 0.19499999284744263

    dat_info['filepath'] = basic_mat.getStr(basic_mat.name.basicMetaData.dat)
    dat_info['n_sample'] = int(basic_mat.getNum(basic_mat.name.basicMetaData.nSample.dat))
    dat_info['sample_rate'] = basic_mat.getNum(basic_mat.name.basicMetaData.SampleRates.dat)
    dat_info['n_ch'] = int(basic_mat.getNum(basic_mat.name.basicMetaData.nCh))
    dat_info['uV_per_bit'] = 0.19499999284744263

    video_info['sample_rate_'] = basic_mat.getNum(basic_mat.name.basicMetaData.SampleRates.video)
    video_info['n_sample'] = int(basic_mat.getNum(basic_mat.name.basicMetaData.video.nFrame))
    video_info['file_path'] = basic_mat.getStr(basic_mat.name.basicMetaData.video.filename)
    video_info['pixel_size'] = [basic_mat.getNum(basic_mat.name.basicMetaData.video.width),
                                basic_mat.getNum(basic_mat.name.basicMetaData.video.height)]

    video_info['state_led_pos'] = tuple(map(tuple, basic_mat.getNum(basic_mat.name.basicMetaData.video.ledRange)))

    detection_range = []
    detection_time = []
    detection_name = []
    with h5py.File(os.path.expanduser('~/data/Fear/triple/' + rat_name + '/' + rat_name + '.basicMetaData.mat'),
                   'r') as f:
        for ref in basic_mat.getNum(basic_mat.name.basicMetaData.chamber.positionRange):
            detection_range.append(tuple(f[ref][()].T[0]))

        for ref in basic_mat.getNum(basic_mat.name.basicMetaData.chamber.detectionintervals):
            detection_time.append(tuple(f[ref][()].T[0]))

        for ref in basic_mat.getNum(basic_mat.name.basicMetaData.chamber.name):
            detection_name.append(re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                                         re.sub('(.)([A-Z][a-z]+)', r'\1_\2', ''.join(map(chr, f[ref])))).lower())
    video_info['behavior_session_pos'] = tuple(detection_range)
    video_info['behavior_session_time'] = tuple(detection_time)
    video_info['behavior_session_name'] = tuple(detection_name)

    ch_name = basic_mat.getStr(basic_mat.name.basicMetaData.Ch.names)
    ch_name = [reg.replace(' cont', '') for reg in ch_name]
    ch_name = [reg.replace('CG', 'Cg') for reg in ch_name]
    ch_name = [reg.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23') for reg in ch_name]
    ch_name = [reg.replace(' L5', 'L5') for reg in ch_name]
    ch_name = [reg.replace('PrLL', 'PL') for reg in ch_name]
    ch_name = [reg.replace('CgL', 'Cg') for reg in ch_name]
    ch_name = [reg.replace('M2L', 'M2_') for reg in ch_name]
    ch_name = [reg.replace('zonaincerta', 'zona_incerta') for reg in ch_name]
    ch_name = [reg.replace('/', '').replace(' ', '_').replace('\u3000', '_') for reg in ch_name]

    ch_info['name'] = tuple(ch_name)
    ch_map = []
    with h5py.File(os.path.expanduser('~/data/Fear/triple/' + rat_name + '/' + rat_name + '.basicMetaData.mat'),
                   'r') as f:
        for n in range(len(basic_mat.getNum(basic_mat.name.basicMetaData.chMap))):
            ch_map.append(tuple(map(int, (f[basic_mat.getNum(basic_mat.name.basicMetaData.chMap)[n]][()].T[0] - 1))))
    # ch_info['ch_map'] = tuple(ch_map)

    n_ch = int(basic_mat.getNum(basic_mat.name.basicMetaData.nCh))

    sh_list = []
    sh_eahc_list = []
    ch_list = []
    pr_list = []

    n_shank = [0, 0, 0, 0]
    n_shank[1] = sum(list(all(list(n < 64 for n in sh)) for sh in ch_map))
    n_shank[2] = sum(list(all(list(n < 64 * 2 for n in sh)) for sh in ch_map))
    n_shank[3] = sum(list(all(list(n < 64 * 3 for n in sh)) for sh in ch_map))

    offset = []
    pr_index = []
    for n in range(1, 4):
        offset.extend([n_shank[n - 1]] * (n_shank[n] - n_shank[n - 1]))
        pr_index.extend([n - 1] * (n_shank[n] - n_shank[n - 1]))

    for ch in range(n_ch):
        for sh in range(len(ch_map)):
            if ch in ch_map[sh]:
                sh_list.append(int(sh))
                sh_eahc_list.append(int(sh - offset[sh]))
                ch_list.append(int(ch_map[sh].index(ch)))
                pr_list.append(int(pr_index[sh]))
                break
        else:
            sh_list.append(np.nan)
            ch_list.append(np.nan)
            sh_eahc_list.append(np.nan)
            pr_list.append(np.nan)

    ch_info['pr'] = pr_list
    ch_info['sh_total'] = sh_list
    ch_info['sh_within_pr'] = sh_eahc_list
    ch_info['ch_within_sh'] = ch_list

    ses_mat = matHandler(base_mat_path + '.sessions.events.mat')
    beh = ses_mat.getNum(ses_mat.name.sessions.timestamps)
    hc = ses_mat.getNum(ses_mat.name.sessions.homecage)
    is_beh = np.array([True] * beh.shape[1] + [False] * hc.shape[1])

    beh_name = detection_name
    beh_name.extend(['homecage%d' % n for n in range(5)])
    ses_t = np.concatenate((beh, hc), axis=1)
    beh_name = np.array(beh_name)

    order = np.argsort(ses_t[1, :])
    session_info = pd.DataFrame()
    session_info['name'] = beh_name[order]
    session_info['start_t'] = ses_t[0, order]
    session_info['end_t'] = ses_t[1, order]
    session_info['is_behavior_session'] = is_beh[order]

    rat_info = {}
    surgery_info = {}

    rat_info['name'] = basic_mat.getStr(basic_mat.name.basicMetaData.Animal.Name)
    temp = basic_mat.getStr(basic_mat.name.basicMetaData.Animal.Info.DateOfBirth)
    rat_info['date_of_birth'] = '%s-%s-%s' % (temp[0:4], temp[4:6], temp[6:8])
    temp = animal_mat.getStr(animal_mat.name.AnimalMetadata.Surgery.Date)
    rat_info['date_of_surgery'] = '%s-%s-%s' % (temp[0:4], temp[4:6], temp[6:8])
    temp = rat_name[-6:]
    rat_info['date_of_experiment'] = '20%s-%s-%s' % (temp[0:2], temp[2:4], temp[4:6])
    rat_info['wight_at_surgery'] = basic_mat.getNum(basic_mat.name.basicMetaData.Animal.Info.WeightGramsAtSurgery)
    rat_info['probe_type'] = animal_mat.getStr(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ProbeLayoutFilenames)
    rat_info['sex'] = basic_mat.getStr(basic_mat.name.basicMetaData.Animal.Info.Sex)
    rat_info['strain'] = basic_mat.getStr(basic_mat.name.basicMetaData.Animal.Info.Strain)
    # rat_info['species']=basic_mat.getStr(bmd.Animal.Info.Species)

    temp = animal_mat.getStr(animal_mat.name.AnimalMetadata.Surgery.Date)
    surgery_info['date_of_surgery'] = '%s-%s-%s' % (temp[0:4], temp[4:6], temp[6:8])
    surgery_info['wight_at_surgery'] = basic_mat.getNum(basic_mat.name.basicMetaData.Animal.Info.WeightGramsAtSurgery)
    surgery_info['target_region'] = animal_mat.getStr(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.TargetRegions)
    surgery_info['hemisphere'] = animal_mat.getStr(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.TargetHemisphere)
    surgery_info['probe_type'] = animal_mat.getStr(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ProbeLayoutFilenames)
    surgery_info['probe_angle_ML'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ImplantAngle.Mediolateral)
    surgery_info['probe_angle_AP'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ImplantAngle.Anteroposterior)
    surgery_info['probe_orientation'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.OrientationOfProbe.FirstGroupRelativeToLastGroupClockwiseDegreesAnteriorIsZero)
    surgery_info['coordinate_AP'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ImplantCoordinates.Anteroposterior)
    surgery_info['coordinate_ML'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ImplantCoordinates.Mediolateral)
    surgery_info['coordinate_DV'] = animal_mat.getNum(
        animal_mat.name.AnimalMetadata.ExtracellEphys.Probes.ImplantCoordinates.DepthFromSurface)

    params = {'matlab_filename': matlab_filename}

    metadata = {'contents': (
        'metadata', 'basic_info', 'session_info', 'ch_info', 'lfp_info', 'dat_info', 'video_info', 'rat_info',
        'surgery_info', 'params'),
        'generator': __name__,
        'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump(
        (metadata, basic_info, session_info, ch_info, lfp_info, dat_info, video_info, rat_info, surgery_info, params),
        base_dir + rat_name + '-basic_info.joblib',
        compress=True)


# %%
def conv_ok_spikes(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = [rat_name + '.okUnit.spikes.mat',
                       'analyses/' + rat_name + '-okUnit.cellInfo.mat',
                       'analyses/' + rat_name + '-clusterStats.mat']

    spk_mat = matHandler(base_mat_path + '.okUnit.spikes.mat')
    cluinfo_mat_path = analyses_mat_path + '-okUnit.cellInfo.mat'
    cluinfo_mat = matHandler(cluinfo_mat_path)

    quality_mat_path = analyses_mat_path + '-clusterStats.mat'
    quality_mat = matHandler(quality_mat_path)

    spiketime = list(spk_mat.getNum(spk_mat.name.okUnit.spikeTime))
    cluster = list(map(int, spk_mat.getNum(spk_mat.name.okUnit.cluster) - 1))

    spikes = pd.DataFrame({'spiketime': spiketime, 'cluster': cluster})

    cluster_info = pd.DataFrame()

    reg_name = list(spk_mat.getStr(spk_mat.name.okUnit.cluInfo.region))
    reg_name = [reg.replace(' cont', '') for reg in reg_name]
    reg_name = [reg.replace('CG', 'Cg') for reg in reg_name]
    reg_name = [reg.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23') for reg in reg_name]
    reg_name = [reg.replace(' L5', 'L5') for reg in reg_name]
    reg_name = [reg.replace('PrLL', 'PL') for reg in reg_name]
    reg_name = [reg.replace('CgL', 'Cg') for reg in reg_name]
    reg_name = [reg.replace('M2L', 'M2_') for reg in reg_name]
    reg_name = [reg.replace('zonaincerta', 'zona_incerta') for reg in reg_name]
    reg_name = [reg.replace('/', '').replace(' ', '_').replace('\u3000', '_') for reg in reg_name]

    cluster_info['channel'] = list(map(int, spk_mat.getNum(spk_mat.name.okUnit.cluInfo.channel) - 1))
    cluster_info['phy_id'] = list(map(int, spk_mat.getNum(spk_mat.name.okUnit.cluInfo.phyID)))
    cluster_info['probe'] = list(map(int, spk_mat.getNum(spk_mat.name.okUnit.cluInfo.probe) - 1))
    cluster_info['shank'] = list(map(int, spk_mat.getNum(spk_mat.name.okUnit.cluInfo.shank) - 1))
    cluster_info['region'] = reg_name
    typeCode = ['inh', 'nc', 'ex']
    cluster_info['type'] = [typeCode[n] for n in
                            map(int, cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.cellType.type) + 1)]

    unit_info = pd.DataFrame(cluster_info)

    f = h5py.File(os.path.expanduser(cluinfo_mat_path), 'r')
    waveform = pd.DataFrame()
    waveform['mean'] = [f[res][()] for res in cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.wave.mean)]
    waveform['std'] = [f[res][()] for res in cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.wave.std)]

    waveform['rise'] = list(cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.rise))
    waveform['decay'] = list(cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.decay))
    waveform['halfwidth'] = list(cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.halfwidth))
    waveform['peak_trough_amp'] = list(cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.peakTroughAmp))
    waveform['is_positive'] = list(cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.waveform.positiveSpike) == 1)

    org_id = list(map(int, cluinfo_mat.getNum(cluinfo_mat.name.okUnitInfo.originalID) - 1))

    quality = pd.DataFrame()

    quality['fr'] = list(quality_mat.getNum(quality_mat.name.clusterStats.fr)[org_id])
    quality['isolation_distance'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isoDist)[org_id])
    quality['isi_index'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isiIdx)[org_id])
    quality['l_ratio'] = list(quality_mat.getNum(quality_mat.name.clusterStats.Lratio)[org_id])
    quality['contamination_rate'] = list(quality_mat.getNum(quality_mat.name.clusterStats.contamRate)[org_id])
    quality['acg'] = list(quality_mat.getNum(quality_mat.name.clusterStats.acg)[:, org_id].T)
    quality['isi_histogram'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isiHist)[:, org_id].T)

    params = {
        'maxContamRate': spk_mat.getNum(spk_mat.name.okUnit.param.maxContamRate),
        'maxIsiIdx': spk_mat.getNum(spk_mat.name.okUnit.param.maxIsiIdx),
        'minAmp': spk_mat.getNum(spk_mat.name.okUnit.param.minAmp),
        'minFR': spk_mat.getNum(spk_mat.name.okUnit.param.minFR),
        'minIsoDist': spk_mat.getNum(spk_mat.name.okUnit.param.minIsoDist),
        'generator_matlab': [spk_mat.getStr(spk_mat.name.okUnit.generator),
                             cluinfo_mat.getStr(cluinfo_mat.name.okUnitInfo.generator),
                             quality_mat.getStr(quality_mat.name.clusterStats.generatore)],
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'spikes', 'cluster_info', 'waveform', 'quality', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, spikes, cluster_info, waveform, quality, params), base_dir + rat_name + '-ok_spikes.joblib',
                compress=True)


# %%
def conv_ng_spikes(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = [rat_name + '.ngUnit.spikes.mat',
                       'analyses/' + rat_name + '-ngUnit.cellInfo.mat',
                       'analyses/' + rat_name + '-clusterStats.mat']
    spk_mat = matHandler(base_mat_path + '.ngUnit.spikes.mat')
    cluinfo_mat_path = analyses_mat_path + '-ngUnit.cellInfo.mat'
    cluinfo_mat = matHandler(cluinfo_mat_path)

    quality_mat_path = analyses_mat_path + '-clusterStats.mat'
    quality_mat = matHandler(quality_mat_path)

    spiketime = list(spk_mat.getNum(spk_mat.name.ngUnit.spikeTime))
    cluster = list(map(int, spk_mat.getNum(spk_mat.name.ngUnit.cluster) - 1))

    spikes = pd.DataFrame({'spiketime': spiketime, 'cluster': cluster})

    cluster_info = pd.DataFrame()

    reg_name = list(spk_mat.getStr(spk_mat.name.ngUnit.cluInfo.region))
    reg_name = [reg.replace(' cont', '') for reg in reg_name]
    reg_name = [reg.replace('CG', 'Cg') for reg in reg_name]
    reg_name = [reg.replace(' L1', 'L23').replace(' L2', 'L23').replace(' L3', 'L23') for reg in reg_name]
    reg_name = [reg.replace(' L5', 'L5') for reg in reg_name]
    reg_name = [reg.replace('PrLL', 'PL') for reg in reg_name]
    reg_name = [reg.replace('CgL', 'Cg') for reg in reg_name]
    reg_name = [reg.replace('M2L', 'M2_') for reg in reg_name]
    reg_name = [reg.replace('zonaincerta', 'zona_incerta') for reg in reg_name]
    reg_name = [reg.replace('/', '').replace(' ', '_').replace('\u3000', '_') for reg in reg_name]

    cluster_info['channel'] = list(map(int, spk_mat.getNum(spk_mat.name.ngUnit.cluInfo.channel) - 1))
    cluster_info['phy_id'] = list(map(int, spk_mat.getNum(spk_mat.name.ngUnit.cluInfo.phyID)))
    cluster_info['probe'] = list(map(int, spk_mat.getNum(spk_mat.name.ngUnit.cluInfo.probe) - 1))
    cluster_info['shank'] = list(map(int, spk_mat.getNum(spk_mat.name.ngUnit.cluInfo.shank) - 1))
    cluster_info['region'] = reg_name

    f = h5py.File(os.path.expanduser(cluinfo_mat_path), 'r')
    waveform = pd.DataFrame()
    waveform['mean'] = [f[res][()] for res in cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.wave.mean)]
    waveform['std'] = [f[res][()] for res in cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.wave.std)]

    waveform['rise'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.rise))
    waveform['decay'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.decay))
    waveform['halfwidth'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.halfwidth))
    waveform['peak_trough_amp'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.peakTroughAmp))
    waveform['is_positive'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.positiveSpike) == 1)

    org_id = list(map(int, cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.originalID) - 1))

    quality = pd.DataFrame()

    quality['fr'] = list(quality_mat.getNum(quality_mat.name.clusterStats.fr)[org_id])
    quality['isolation_distance'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isoDist)[org_id])
    quality['isi_index'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isiIdx)[org_id])
    quality['l_ratio'] = list(quality_mat.getNum(quality_mat.name.clusterStats.Lratio)[org_id])
    quality['contamination_rate'] = list(quality_mat.getNum(quality_mat.name.clusterStats.contamRate)[org_id])
    quality['acg'] = list(quality_mat.getNum(quality_mat.name.clusterStats.acg)[:, org_id].T)
    quality['isi_histogram'] = list(quality_mat.getNum(quality_mat.name.clusterStats.isiHist)[:, org_id].T)
    quality['peak_trough_amp'] = list(cluinfo_mat.getNum(cluinfo_mat.name.ngUnitInfo.waveform.peakTroughAmp))

    params = {
        'maxContamRate': spk_mat.getNum(spk_mat.name.ngUnit.param.maxContamRate),
        'maxIsiIdx': spk_mat.getNum(spk_mat.name.ngUnit.param.maxIsiIdx),
        'minAmp': spk_mat.getNum(spk_mat.name.ngUnit.param.minAmp),
        'minFR': spk_mat.getNum(spk_mat.name.ngUnit.param.minFR),
        'minIsoDist': spk_mat.getNum(spk_mat.name.ngUnit.param.minIsoDist),
        'generator_matlab': [spk_mat.getStr(spk_mat.name.ngUnit.generator),
                             cluinfo_mat.getStr(cluinfo_mat.name.ngUnitInfo.generator),
                             quality_mat.getStr(quality_mat.name.clusterStats.generatore)],
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'spikes', 'cluster_info', 'waveform', 'quality', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, spikes, cluster_info, waveform, quality, params), base_dir + rat_name + '-ng_spikes.joblib',
                compress=True)


# %%
def conv_sleep_states(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = rat_name + '.sleepScore.mat'
    slp_mat = matHandler(base_mat_path + '.sleepScore.mat')

    state_name = [n.lower() for n in slp_mat.getStr(slp_mat.name.sleepScore.stateName)]
    ts = slp_mat.getNum(slp_mat.name.sleepScore.usual)
    sleep_states = pd.DataFrame()
    sleep_states['start_t'] = ts[0, :]
    sleep_states['end_t'] = ts[1, :]
    sleep_states['state'] = [state_name[int(n - 1)] for n in ts[2, :]]

    sleep_packets = pd.DataFrame()
    ts = slp_mat.getNum(slp_mat.name.sleepScore.withMA)
    sleep_packets['start_t'] = ts[0, :]
    sleep_packets['end_t'] = ts[1, :]
    sleep_packets['state'] = [state_name[int(n - 1)] for n in ts[2, :]]

    params = {
        'slowave_ch': int(slp_mat.getNum(slp_mat.name.sleepScore.param.ch.slowwave) - 1),
        'theta_ch': int(slp_mat.getNum(slp_mat.name.sleepScore.param.ch.theta) - 1),
        'max_ma_duration': slp_mat.getNum(slp_mat.name.sleepScore.param.maThreshold),
        'generator_matlab': slp_mat.getStr(slp_mat.name.sleepScore.generator),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'sleep_states', 'sleep_packets', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, sleep_states, sleep_packets, params), base_dir + rat_name + '-sleep_states.joblib',
                compress=True)


# %%
def conv_cues(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    cue_mat = matHandler(base_mat_path + '.cues.events.mat')
    matlab_filename = rat_name + '.cues.events.mat'

    cue_time = cue_mat.getNum(cue_mat.name.cues.timestamps.Pip)
    ses_mat = matHandler(base_mat_path + '.sessions.events.mat')
    beh = ses_mat.getNum(ses_mat.name.sessions.timestamps)

    beh_idx = -1 * np.ones(cue_time.shape[1], dtype='int')
    cue_grp = -1 * np.ones(cue_time.shape[1], dtype='int')
    cue_cnt = -1 * np.ones(cue_time.shape[1], dtype='int')
    for n in range(beh.shape[1]):
        beh_idx[(cue_time[0, :] > beh[0, n]) & (cue_time[0, :] < beh[1, n])] = n
        cue_subset = cue_time[0, (cue_time[0, :] > beh[0, n]) & (cue_time[0, :] < beh[1, n])]
        cue_offsets = np.argwhere(np.diff(cue_subset) > 10)
        cue_grp_subset = np.zeros_like(cue_subset, dtype='int')
        cue_cnt_subset = np.zeros_like(cue_subset, dtype='int')
        for m in cue_offsets:
            cue_grp_subset[m[0] + 1:] += 1
        cue_grp[(cue_time[0, :] > beh[0, n]) & (cue_time[0, :] < beh[1, n])] = cue_grp_subset

        for m in range(1, len(cue_cnt_subset)):
            if cue_grp_subset[m - 1] == cue_grp_subset[m]:
                cue_cnt_subset[m] = cue_cnt_subset[m - 1] + 1
        cue_cnt[(cue_time[0, :] > beh[0, n]) & (cue_time[0, :] < beh[1, n])] = cue_cnt_subset

    cues = pd.DataFrame()
    cues['start_t'] = cue_time[0, :]
    cues['end_t'] = cue_time[1, :]
    cues['session_idx'] = beh_idx
    cues['train_idx'] = cue_grp
    cues['pip_idx'] = cue_cnt

    params = {
        'ch': list(map(int, cue_mat.getNum(cue_mat.name.cues.detectorinfo.detectionparms.chList) - 1)),
        'evt_filename': os.path.basename(cue_mat.getStr(cue_mat.name.cues.detectorinfo.detectionparms.evtFileName)),
        'minDuration': cue_mat.getNum(cue_mat.name.cues.detectorinfo.detectionparms.minDuration),
        'minInterval': cue_mat.getNum(cue_mat.name.cues.detectorinfo.detectionparms.minInterval),
        'nBaseline': cue_mat.getNum(cue_mat.name.cues.detectorinfo.detectionparms.nBaseline),
        'threshold': cue_mat.getNum(cue_mat.name.cues.detectorinfo.detectionparms.threshold),
        'generator_matlab': cue_mat.getStr(cue_mat.name.cues.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}
    metadata = {'contents': ('metadata', 'cues', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, cues, params), base_dir + rat_name + '-cues.joblib',
                compress=True)


# %%
def conv_shocks(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = rat_name + '.shocks.events.mat'
    shk_mat = matHandler(base_mat_path + '.shocks.events.mat')

    sh_l = shk_mat.getNum(shk_mat.name.shocks.timestamps.ShockL)
    sh_r = shk_mat.getNum(shk_mat.name.shocks.timestamps.ShockR)
    sh = np.hstack((sh_l, sh_r))
    lr = ['left'] * sh_l.shape[1] + ['right'] * sh_r.shape[1]

    order = np.argsort(sh[1, :])
    lr = [lr[n] for n in order]
    sh = sh[:, order]

    sh_offsets = np.argwhere(np.diff(sh[0, :]) > 10)
    sh_grp = np.zeros((sh.shape[1]), dtype=np.int8)
    for n in sh_offsets:
        sh_grp[n[0] + 1:] += 1
    sh_cnt = np.zeros((sh.shape[1]), dtype=np.int8)
    for n in range(1, len(sh_cnt)):
        if sh_grp[n - 1] == sh_grp[n]:
            sh_cnt[n] = sh_cnt[n - 1] + 1

    shocks = pd.DataFrame()
    shocks['start_t'] = sh[0, :]
    shocks['end_t'] = sh[1, :]
    shocks['train_idx'] = sh_grp
    shocks['pulse_idx'] = sh_cnt
    shocks['l/r'] = lr

    params = {
        'ch': list(map(int, shk_mat.getNum(shk_mat.name.shocks.detectorinfo.detectionparms.chList) - 1)),
        'evt_filename': os.path.basename(shk_mat.getStr(shk_mat.name.shocks.detectorinfo.detectionparms.evtFileName)),
        'minDuration': shk_mat.getNum(shk_mat.name.shocks.detectorinfo.detectionparms.minDuration),
        'minInterval': shk_mat.getNum(shk_mat.name.shocks.detectorinfo.detectionparms.minInterval),
        'nBaseline': shk_mat.getNum(shk_mat.name.shocks.detectorinfo.detectionparms.nBaseline),
        'threshold': shk_mat.getNum(shk_mat.name.shocks.detectorinfo.detectionparms.threshold),
        'generator_matlab': shk_mat.getStr(shk_mat.name.shocks.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'shocks', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, shocks, params), base_dir + rat_name + '-shocks.joblib',
                compress=True)


# %%
def conv_videoframe(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = rat_name + '.videoFrames.events.mat'
    vf_mat = matHandler(base_mat_path + '.videoFrames.events.mat')
    video_ttl = vf_mat.getNum(vf_mat.name.videoFrames.timestamps)
    params = {
        'ch': int(vf_mat.getNum(vf_mat.name.videoFrames.detectorinfo.detectionparms.chList) - 1),
        'evt_filename': os.path.basename(
            vf_mat.getStr(vf_mat.name.videoFrames.detectorinfo.detectionparms.evtFileName)),
        'minDuration': vf_mat.getNum(vf_mat.name.videoFrames.detectorinfo.detectionparms.minDuration),
        'minInterval': vf_mat.getNum(vf_mat.name.videoFrames.detectorinfo.detectionparms.minInterval),
        'nBaseline': vf_mat.getNum(vf_mat.name.videoFrames.detectorinfo.detectionparms.nBaseline),
        'threshold': vf_mat.getNum(vf_mat.name.videoFrames.detectorinfo.detectionparms.threshold),
        'generator_matlab': vf_mat.getStr(vf_mat.name.videoFrames.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'video_ttl', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, video_ttl, params), base_dir + rat_name + '-video_ttl.joblib',
                compress=True)


# %%
def conv_clock(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = rat_name + '.clocks.events.mat'
    clk_mat = matHandler(base_mat_path + '.clocks.events.mat')
    clock_ttl = clk_mat.getNum(clk_mat.name.clocks.timestamps)

    params = {
        'ch': int(clk_mat.getNum(clk_mat.name.clocks.detectorinfo.detectionparms.chList) - 1),
        'evt_filename': os.path.basename(clk_mat.getStr(clk_mat.name.clocks.detectorinfo.detectionparms.evtFileName)),
        'minDuration': clk_mat.getNum(clk_mat.name.clocks.detectorinfo.detectionparms.minDuration),
        'minInterval': clk_mat.getNum(clk_mat.name.clocks.detectorinfo.detectionparms.minInterval),
        'nBaseline': clk_mat.getNum(clk_mat.name.clocks.detectorinfo.detectionparms.nBaseline),
        'threshold': clk_mat.getNum(clk_mat.name.clocks.detectorinfo.detectionparms.threshold),
        'generator_matlab': clk_mat.getStr(clk_mat.name.clocks.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}
    metadata = {'contents': ('metadata', 'clock_ttl', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, clock_ttl, params), base_dir + rat_name + '-clock_ttl.joblib',
                compress=True)


# %%
def conv_heartbeat(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)

    matlab_filename = rat_name + '.heartBeat.events.mat'
    hb_mat = matHandler(base_mat_path + '.heartBeat.events.mat')

    heartbeat = hb_mat.getNum(hb_mat.name.heartBeat.timestamps)

    params = {
        'median_filter_order': int(hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.medFiltOrder)),
        'ecg_ch': int(hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.ecgCh) - 1),
        'evt_filename': os.path.basename(hb_mat.getStr(hb_mat.name.heartBeat.detectorinfo.detectionparms.evtFileName)),
        'min_peak': hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.minPeak),
        'min_peak_distance': hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.minPeakDistance),
        'n_baseline': hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.nBaseline),
        'threshold': hb_mat.getNum(hb_mat.name.heartBeat.detectorinfo.detectionparms.threshold),
        'generator_matlab': hb_mat.getStr(hb_mat.name.heartBeat.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'heartbeat', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, heartbeat, params), base_dir + rat_name + '-heartbeat.joblib',
                compress=True)


# %%
def conv_freeze(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.freezeHMM.events.mat'

    frz_mat = matHandler(base_mat_path + '.freezeHMM.events.mat')
    frz = frz_mat.getNum(frz_mat.name.freezeHMM.timestamps)
    freeze = pd.DataFrame()
    freeze['start_t'] = frz[0, :]
    freeze['end_t'] = frz[1, :]

    params = {
        'ob_deltaband': frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.OBdeltaBand),
        'ob_thetaband': frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.OBthetaBand),
        'min_freeze_duration': frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.minFreezeDuration),
        'min_interval': frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.minInterval),
        'min_wake_duration': frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.minWakeDuration),
        'n_state': int(frz_mat.getNum(frz_mat.name.freezeHMM.detectorinfo.detectionparam.nState)),
        'evt_filename': os.path.basename(
            frz_mat.getStr(frz_mat.name.freezeHMM.detectorinfo.detectionparam.evtFileName)),
        'generator_matlab': frz_mat.getStr(frz_mat.name.freezeHMM.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'freeze', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, freeze, params), base_dir + rat_name + '-freeze.joblib',
                compress=True)


# %%
def conv_hfo(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = [rat_name + '.amyHFO.events.mat'
                                  'analyses/' + rat_name + '-hfoPeak.mat']

    hfo_mat = matHandler(base_mat_path + '.amyHFO.events.mat')
    hfo_freq_mat = matHandler(analyses_mat_path + '-hfoPeak.mat')

    ts = hfo_mat.getNum(hfo_mat.name.amyHFO.timestamps)

    power = hfo_mat.getNum(hfo_mat.name.amyHFO.peaks.power)
    ptime = hfo_mat.getNum(hfo_mat.name.amyHFO.peaks.timestamps)
    pfreq = hfo_freq_mat.getNum((hfo_freq_mat.name.hfoPeak.freq))
    sh = (hfo_mat.getNum(hfo_mat.name.amyHFO.peaks.sh) - 1).astype('int')

    hfo = pd.DataFrame({'start_t': ts[0, :],
                        'end_t': ts[1, :],
                        'peak_t': ptime,
                        'peak_power': power,
                        'peak_frequency': pfreq,
                        'peak_shank': sh})

    params = {
        'evt_filename': rat_name + '.HFO.evt',
        'probe_region': hfo_mat.getStr(hfo_mat.name.amyHFO.region),
        'baseline_frame': hfo_mat.getNum(hfo_mat.name.amyHFO.param.baselineFrame),
        'exclude_frame': hfo_mat.getNum(hfo_mat.name.amyHFO.param.excludeFrame),
        'filter_order': int(hfo_mat.getNum(hfo_mat.name.amyHFO.param.filOrder)),
        'frequency_range': hfo_mat.getNum(hfo_mat.name.amyHFO.param.freqRange),
        'high_threshold_factor': hfo_mat.getNum(hfo_mat.name.amyHFO.param.highThresholdFactor),
        'low_threshold_factor': hfo_mat.getNum(hfo_mat.name.amyHFO.param.lowThresholdFactor),
        'max_duration': hfo_mat.getNum(hfo_mat.name.amyHFO.param.maxDuration),
        'min_duration': hfo_mat.getNum(hfo_mat.name.amyHFO.param.minDuration),
        'min_inter_event_interval': hfo_mat.getNum(hfo_mat.name.amyHFO.param.minInterEventInterval),
        'smoothing_window': hfo_mat.getNum(hfo_mat.name.amyHFO.param.smoothingWindow),
        'generator_matlab': [hfo_mat.getStr(hfo_mat.name.amyHFO.generator),
                             hfo_freq_mat.getStr(hfo_freq_mat.name.hfoPeak.generator)],
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'hfo', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, hfo, params), base_dir + rat_name + '-amygdalar_hfo.joblib',
                compress=True)


# %%
def conv_swr(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.ripples.events.mat'

    if not os.path.isfile(os.path.expanduser(base_mat_path + '.ripples.events.mat')):
        print('\t\t skipped: {fname} is not found'.format(fname=matlab_filename))
        return

    swr_mat = matHandler(base_mat_path + '.ripples.events.mat')
    ts = swr_mat.getNum(swr_mat.name.ripples.timestamps)
    peak_t = swr_mat.getNum(swr_mat.name.ripples.peaks.timestamps)
    power = swr_mat.getNum(swr_mat.name.ripples.peaks.power)
    negative = swr_mat.getNum(swr_mat.name.ripples.peaks.negative)
    sh = (swr_mat.getNum(swr_mat.name.ripples.peaks.sh) - 1).astype('int')
    ch = (swr_mat.getNum(swr_mat.name.ripples.peaks.ch) - 1).astype('int')

    swr = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': peak_t,
        'negative_t': negative,
        'peak_power': power,
        'max_sh': sh,
        'max_ch': ch
    })

    ch_list = []
    with h5py.File(os.path.expanduser(base_mat_path + '.ripples.events.mat'), 'r') as f:
        for ref in swr_mat.getNum(swr_mat.name.ripples.detectorinfo.chList):
            ch_list.append((f[ref][()] - 1).astype('int').reshape(-1))

    params = {
        'ch_list': ch_list,
        'baseline_frame': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.baselineFrame).astype('int'),
        'evt_filename': os.path.basename(swr_mat.getStr(swr_mat.name.ripples.detectorinfo.detectionparms.evtFileName)),
        'exclude_frame': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.excludeFrame).astype('int'),
        'filter_order': int(swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.filOrder)),
        'frame_range': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.frameRange).astype('int'),
        'ripple_frequency_range': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.freqRange),
        'ripple_high_threshold_factor': swr_mat.getNum(
            swr_mat.name.ripples.detectorinfo.detectionparms.highThresholdFactor),
        'ripple_low_threshold_factor': swr_mat.getNum(
            swr_mat.name.ripples.detectorinfo.detectionparms.lowThresholdFactor),
        'max_ripple_duration': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.maxRippleDuration),
        'min_ch_fraction': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.minChFrac),
        'min_inter_ripple_interval': swr_mat.getNum(
            swr_mat.name.ripples.detectorinfo.detectionparms.minInterRippleInterval),
        'min_ripple_duration': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.minRippleDuration),
        'smoothing_window': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.smoothingWindow),
        'superficial_top': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.superficialTop).astype(
            'bool'),
        'sharp-wave_duration': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.swDuration),
        'sharp-wave_filter_frequency': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.swFilFreq),
        'sharp-wave_threshold': swr_mat.getNum(swr_mat.name.ripples.detectorinfo.detectionparms.swThreshold),
        'generator_matlab': swr_mat.getStr(swr_mat.name.ripples.detectorinfo.detectorname),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'swr', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, swr, params), base_dir + rat_name + '-hippocampal_swr.joblib',
                compress=True)


# %%
def conv_pfc_ripples(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcRipple.events.mat'
    pfc_mat = matHandler(base_mat_path + '.pfcRipple.events.mat')

    ts = pfc_mat.getNum(pfc_mat.name.pfcRipple.timestamps)
    pfc_ripples = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcRipple.peaks.timestamps),
        'peak_power': pfc_mat.getNum(pfc_mat.name.pfcRipple.peaks.power),
        'peak_ch': (pfc_mat.getNum(pfc_mat.name.pfcRipple.peaks.sh) - 1).astype('int')
    })

    params = {
        'evt_filename': rat_name + '.pRp.evt',
        'baseline_frame': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.baselineFrame).astype('int'),
        'exclude_frame': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.baselineFrame).astype('int'),
        'filter_order': int(pfc_mat.getNum(pfc_mat.name.pfcRipple.param.filOrder)),
        'frequency_range': (pfc_mat.getNum(pfc_mat.name.pfcRipple.freqRange)),
        'high_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.highThresholdFactor),
        'low_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.lowThresholdFactor),
        'max_duration': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.maxDuration),
        'min_duration': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.minDuration),
        'min_inter_event_interval': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.minInterEventInterval),
        'smoothing_window': pfc_mat.getNum(pfc_mat.name.pfcRipple.param.smoothingWindow),
        'generator_matlab': [pfc_mat.getStr(pfc_mat.name.pfcRipple.generator),
                             'fear_detectPfcGamma'],
        'matlab_filename': matlab_filename}
    metadata = {'contents': ('metadata', 'pfc_ripples', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_ripples, params), base_dir + rat_name + '-prefrontal_ripples.joblib',
                compress=True)


# %%
def conv_pfc_fast_gamma(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcFastGamma.events.mat'
    pfc_mat = matHandler(base_mat_path + '.pfcFastGamma.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.pfcFastGamma.timestamps)

    pfc_fast_gamma = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.peaks.timestamps),
        'peak_power': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.peaks.power),
        'peak_ch': (pfc_mat.getNum(pfc_mat.name.pfcFastGamma.peaks.sh) - 1).astype('int')
    })
    params = {
        'evt_filename': rat_name + '.pFg.evt',
        'baseline_frame': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.baselineFrame).astype('int'),
        'exclude_frame': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.baselineFrame).astype('int'),
        'filter_order': int(pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.filOrder)),
        'frequency_range': (pfc_mat.getNum(pfc_mat.name.pfcFastGamma.freqRange)),
        'high_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.highThresholdFactor),
        'low_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.lowThresholdFactor),
        'max_duration': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.maxDuration),
        'min_duration': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.minDuration),
        'min_inter_event_interval': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.minInterEventInterval),
        'smoothing_window': pfc_mat.getNum(pfc_mat.name.pfcFastGamma.param.smoothingWindow),
        'generator_matlab': [pfc_mat.getStr(pfc_mat.name.pfcFastGamma.generator),
                             'fear_detectPfcGamma'],
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_fast_gamma', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_fast_gamma, params), base_dir + rat_name + '-prefrontal_fast_gamma.joblib',
                compress=True)


# %%
def conv_pfc_slow_gamma(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcSlowGamma.events.mat'
    pfc_mat = matHandler(base_mat_path + '.pfcSlowGamma.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.timestamps)

    pfc_slow_gamma = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.peaks.timestamps),
        'peak_power': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.peaks.power),
        'peak_ch': (pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.peaks.sh) - 1).astype('int')
    })
    params = {
        'evt_filename': rat_name + '.pSg.evt',
        'baseline_frame': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.baselineFrame).astype('int'),
        'exclude_frame': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.baselineFrame).astype('int'),
        'filter_order': int(pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.filOrder)),
        'frequency_range': (pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.freqRange)),
        'high_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.highThresholdFactor),
        'low_threshold_factor': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.lowThresholdFactor),
        'max_duration': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.maxDuration),
        'min_duration': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.minDuration),
        'min_inter_event_interval': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.minInterEventInterval),
        'smoothing_window': pfc_mat.getNum(pfc_mat.name.pfcSlowGamma.param.smoothingWindow),
        'generator_matlab': [pfc_mat.getStr(pfc_mat.name.pfcSlowGamma.generator),
                             'fear_detectPfcLowGamma'],
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_slow_gamma', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_slow_gamma, params), base_dir + rat_name + '-prefrontal_slow_gamma.joblib',
                compress=True)


# %%
def conv_pfc_slow_wave(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcSlowWave.new.events.mat'
    pfc_mat = matHandler(base_mat_path + '.pfcSlowWave.new.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.pfcSlowWave.timestamps)
    slope = pfc_mat.getNum(pfc_mat.name.pfcSlowWave.slope)

    pfc_slow_wave = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.peak.timestamps),
        'peak_power': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.peak.amplitude),
        'upward_slope': slope[0, :],
        'downward_slope': slope[1, :]
    })
    params = {
        'ch_list': (pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.chList) - 1).astype('int').reshape(-1),
        'duration': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.duration),
        'evt_filename': os.path.basename(pfc_mat.getStr(pfc_mat.name.pfcSlowWave.param.evtFileName)),
        'filter_order': int(pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.filtOrder)),
        'frequency_range': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.freq),
        'min_peak': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.minPeak),
        'min_trough': pfc_mat.getNum(pfc_mat.name.pfcSlowWave.param.minTrough),
        'generator_matlab': pfc_mat.getStr(pfc_mat.name.pfcSlowWave.detector),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_slow_wave', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_slow_wave, params), base_dir + rat_name + '-prefrontal_slow_wave.joblib',
                compress=True)


# %%
def conv_pfc_k_complex(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.kComplex.events.mat'
    pfc_mat = matHandler(base_mat_path + '.kComplex.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.kComplex.timestamps)
    sw_t = pfc_mat.getNum(pfc_mat.name.kComplex.slowwave.timestamps)
    sp_t = pfc_mat.getNum(pfc_mat.name.kComplex.spindle.timestamps)

    pfc_k_complex = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'slowave_onset_t': sw_t[0, :],
        'slowave_offset_t': sw_t[1, :],
        'spindle_onset_t': sp_t[0, :],
        'spindle_offset_t': sp_t[1, :],
        'slowwave_peak_t': pfc_mat.getNum(pfc_mat.name.kComplex.slowwave.peak),
        'spindle_peak_t': pfc_mat.getNum(pfc_mat.name.kComplex.spindle.peak),
        'slowwave_id': (pfc_mat.getNum(pfc_mat.name.kComplex.slowwave.index) - 1).astype(int),
        'spindle_id': (pfc_mat.getNum(pfc_mat.name.kComplex.spindle.index) - 1).astype(int)})
    params = {
        'tGap': pfc_mat.getNum(pfc_mat.name.kComplex.param.tGap),
        'evt_filename': os.path.basename(pfc_mat.getStr(pfc_mat.name.kComplex.param.evtFileName)),
        'generator_matlab': pfc_mat.getStr(pfc_mat.name.kComplex.generator),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_k_complex', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_k_complex, params), base_dir + rat_name + '-prefrontal_k_complex.joblib',
                compress=True)


# %%
def conv_pfc_spindle(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcSpindle.events.mat'
    pfc_mat = matHandler(base_mat_path + '.pfcSpindle.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.pfcSpindle.timestamps)

    pfc_spindle = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcSpindle.peaktime),
        'trough_t': pfc_mat.getNum(pfc_mat.name.pfcSpindle.troughtime),
        'peak_power': pfc_mat.getNum(pfc_mat.name.pfcSpindle.peakPower)})

    params = {
        'chList': (pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.chList) - 1).astype('int'),
        'evt_filename': os.path.basename(pfc_mat.getStr(pfc_mat.name.pfcSpindle.param.evtFileName)),
        'filter_order': int(pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.filtOrder)),
        'frequency_range': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.freq),
        'min_Duration': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.minDur),
        'min_inter_event_interval': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.minIEI),
        'min_peak': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.minPeak),
        'smooth_sigma': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.smoothSigma),
        'threshold': pfc_mat.getNum(pfc_mat.name.pfcSpindle.param.threhold),
        'generator_matlab': pfc_mat.getStr(pfc_mat.name.pfcSpindle.generator),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_spindle', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_spindle, params), base_dir + rat_name + '-prefrontal_spindle.joblib',
                compress=True)

# %%
def conv_pfc_off(rat_name):
    (base_mat_path, analyses_mat_path, base_dir, analyses_dir, data_root) = get_save_dir(rat_name)
    matlab_filename = rat_name + '.pfcOff.events.mat'

    pfc_mat = matHandler(base_mat_path + '.pfcOff.events.mat')
    ts = pfc_mat.getNum(pfc_mat.name.pfcOff.timestamps)

    if len(ts.shape)<2:
        print('\t\t skipped: OFF in {rat} is not detected'.format(rat=rat_name))
        return

    pfc_off = pd.DataFrame({
        'start_t': ts[0, :],
        'end_t': ts[1, :],
        'peak_t': pfc_mat.getNum(pfc_mat.name.pfcOff.peak.timestamps),
        'peak_depth': pfc_mat.getNum(pfc_mat.name.pfcOff.peak.depth),
    })
    params = {
        'min_duration': pfc_mat.getNum(pfc_mat.name.pfcOff.param.durRange)[0],
        'max_duration': pfc_mat.getNum(pfc_mat.name.pfcOff.param.durRange)[1],
        'spike_binsize': pfc_mat.getNum(pfc_mat.name.pfcOff.param.spkBin),
        'spike_smoothing_sigma': pfc_mat.getNum(pfc_mat.name.pfcOff.param.spkSmSigma),
        'threshold_edge': pfc_mat.getNum(pfc_mat.name.pfcOff.param.threshold),
        'threshold_dip': pfc_mat.getNum(pfc_mat.name.pfcOff.param.dipThreshold),
        'evt_filename': os.path.basename(pfc_mat.getStr(pfc_mat.name.pfcOff.param.evtFileName)),
        'generator_matlab': pfc_mat.getStr(pfc_mat.name.pfcOff.detector),
        'matlab_filename': matlab_filename}

    metadata = {'contents': ('metadata', 'pfc_off', 'params'),
                'generator': __name__,
                'generate_date': datetime.today().strftime('%Y-%m-%d')}

    joblib.dump((metadata, pfc_off, params), base_dir + rat_name + '-prefrontal_off.joblib',
                compress=True)

if __name__ == '__main__':
    main()
