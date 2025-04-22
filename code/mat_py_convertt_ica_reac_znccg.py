#%%
import h5py
import numpy as np
from pathlib import Path
from my_tools import data_location
from datetime import date
import joblib
#%%
py_name = "mat_py_convertt_ica_reac_znccg.py"
rats = data_location.get_rats()
#%%
data_dir = Path('~/data/fear_data/').expanduser()
for rat in rats:
    ccg_file_path = data_dir / f"{rat}-icaReacZNCCG.mat"
    save_dir = Path(f"~/Dropbox/analysesPython/data/{rat}/analyses/").expanduser()
    sig_file_path = data_dir / f"{rat}-icaReacZNCCG_sig.mat"

    with h5py.File(ccg_file_path, 'r') as ccg_file, h5py.File(sig_file_path, 'r') as sig_file:
        # 参照を取得
        nrem = ccg_file['icaReacZNCCG']['nrem']
        for n in range(nrem.shape[0]):
            ccg_value=((ccg_file[nrem[n, 0]]["real"]["ccg"])[:])
            # acg_value=((f[nrem[n, 0]]["real"]["acg"])[:])
            mean=ccg_file[nrem[n, 0]]['mean'][:]
            std=ccg_file[nrem[n, 0]]['std'][:]
            nBin=ccg_file[nrem[n, 0]]['nBin'][:].flatten()
            significance=sig_file[sig_file["icaReacZNCCG_sig"]["nrem"][n,0]]["significance"][:]
            ccg={
                "ccg": ccg_value,
                "mean": mean,
                "std": std,
                "nBin": nBin,
                "significance":significance}


            ci95=ccg_file[nrem[n, 0]]['shuffle']["ci95"][:]
            ci99=ccg_file[nrem[n, 0]]['shuffle']["ci99"][:]
            global95=ccg_file[nrem[n, 0]]['shuffle']["global95"][:]
            global99=ccg_file[nrem[n, 0]]['shuffle']["global99"][:]
            shuffle_mean=ccg_file[nrem[n, 0]]['shuffle']["mean"][:]
            shuffle={
                "ci95": ci95,
                "ci99": ci99,
                "global95": global95,
                "global99": global99,
                "mean": shuffle_mean}

            param_ref=ccg_file[ccg_file['icaReacZNCCG']["param"][n,0]]
            params={}
            for key in param_ref.keys():
                params[key]=param_ref[key][0][0]

            region_ref=ccg_file[ccg_file['icaReacZNCCG']["region"][n,0]]
            regions=[]
            for m in range(region_ref.shape[0]):
                region = "".join([chr(int(x)) for x in ccg_file[region_ref[m,0]][:].flatten()])
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
                regions.append(region)

            template = "".join(map(chr,map(int,ccg_file[ccg_file['icaReacZNCCG']["template"][n,0]][:].flatten())))
            ccg_generator="".join(map(chr, map(int, ccg_file[ccg_file["icaReacZNCCG"]["generator"][n,0]][:].flatten())))
            sig_generator="".join(map(chr, map(int, sig_file[sig_file["icaReacZNCCG_sig"]["generator"][n,0]][:].flatten())))
            reactID = ccg_file[ccg_file['icaReacZNCCG']["instReacID"][n,0]][:].flatten().astype(int)
            pair_id=ccg_file[ccg_file['icaReacZNCCG']['pairID'][n,0]][:].astype(int)

            save_filename = f"{rat}_icaReacZNCCG_{template}.joblib"
        
            params["reactID"]=reactID
            params["region"]=regions
            params["pair_id"]=pair_id
            params["template"]=template
            params["generator_matlab"]=[ccg_generator,sig_generator]
            params["matlab_filename"] = [f"analyses/{rat}-icaReacZNCCG.mat",
                                         f"{rat}-icaReacZNCCG_sig.mat"]
            
            metadata = {'contents': ('metadata', 'ccg', 'shuffle','params'),
                        'generator': py_name,
                        'generate_date': date.today().strftime('%Y-%m-%d')}
            joblib.dump((metadata, ccg, shuffle, params), 
                        save_dir / save_filename, compress=True)



    # ccg = f[f['icaReacZNCCG']['nrem'][0, 0]]
    # param
    # region
    # template
    # generator
    # instReacID
    # 参照を解決して実際のデータにアクセス
    # actual_data = f[ref]
    
    # # データのキーを表示
    # ccg=actual_data['real']['ccg'][:]

# %%
