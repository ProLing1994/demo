from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.utils.hdf5_tools import *
from Basic.utils.folder_tools import *


def calc_stats(cfg, data_spk_pd, hdf5_dir, hdf5_normalize_dir, state_name='stats_jnt.h5'):
    # init
    print("number of utterances = "+str(len(data_spk_pd)))
    stdim = cfg.net.yaml["cycle_vae_params"]["stdim"]

    # feat_org_lf0
    scaler_feat_org_lf0 = StandardScaler()

    # mcep
    mcep_var_range = []

    # f0_range
    f0s_range = np.empty((0))

    # dir
    input_dir = os.path.join(hdf5_dir, 'world')
    output_dir = os.path.join(hdf5_normalize_dir, 'world')
    create_folder(output_dir)
    
    
    for _, row in data_spk_pd.iterrows():
        # init
        wav_fpath = row['file']
        key = row['key']
        key_name = key.split('.')[0]

        print(wav_fpath)

        # feat_org_lf0
        feat_org_lf0 = read_hdf5(os.path.join(input_dir, f"{key_name}.h5"), "feat_org_lf0")
        scaler_feat_org_lf0.partial_fit(feat_org_lf0)

        # mcep
        mcep_range = feat_org_lf0[:, stdim:]
        mcep_var_range.append(np.var(mcep_range, axis=0))

        # f0_range
        f0_range = read_hdf5(os.path.join(input_dir, f"{key_name}.h5"), "f0_range")
        nonzero_indices = np.nonzero(f0_range)
        f0s_range = np.concatenate([f0s_range, f0_range[nonzero_indices]])

    # feat_org_lf0
    mean_feat_org_lf0 = scaler_feat_org_lf0.mean_
    scale_feat_org_lf0 = scaler_feat_org_lf0.scale_
    print(mean_feat_org_lf0)
    print(scale_feat_org_lf0)

    # mcep
    gv_range_mean = np.mean(np.array(mcep_var_range), axis=0)
    gv_range_var = np.var(np.array(mcep_var_range), axis=0)
    print(gv_range_mean)
    print(gv_range_var)
    
    # f0_range
    f0_range_mean = np.mean(f0s_range)
    f0_range_std = np.std(f0s_range)
    print(f0_range_mean)
    print(f0_range_std)
    
    lf0_range_mean = np.mean(np.log(f0s_range))
    lf0_range_std = np.std(np.log(f0s_range))
    print(lf0_range_mean)
    print(lf0_range_std)

    # save hdf5
    write_hdf5(os.path.join(output_dir, state_name), "mean_feat_org_lf0", mean_feat_org_lf0)
    write_hdf5(os.path.join(output_dir, state_name), "scale_feat_org_lf0", scale_feat_org_lf0)
    write_hdf5(os.path.join(output_dir, state_name), "gv_range_mean", gv_range_mean)
    write_hdf5(os.path.join(output_dir, state_name), "gv_range_var", gv_range_var)

    write_hdf5(os.path.join(output_dir, state_name), "f0_range_mean", f0_range_mean)
    write_hdf5(os.path.join(output_dir, state_name), "f0_range_std", f0_range_std)
    write_hdf5(os.path.join(output_dir, state_name), "lf0_range_mean", lf0_range_mean)
    write_hdf5(os.path.join(output_dir, state_name), "lf0_range_std", lf0_range_std)

    return 


def preprocess_audio_hdf5_normalize_state_normal(cfg, data_pd, hdf5_dir, hdf5_normalize_dir):
    # init 
    spk_list = data_pd['speaker'].to_list()
    spk_list = list(set(spk_list))
    spk_list.sort()

    # speaker normalize state
    for idx in tqdm(range(len(spk_list))):
        spk_name = spk_list[idx]
        data_spk_pd = data_pd[data_pd['speaker'] == spk_name]
        calc_stats(cfg, data_spk_pd, hdf5_dir, hdf5_normalize_dir, f"stats_spk_{spk_name}.h5")

    # joint normalize state
    calc_stats(cfg, data_pd, hdf5_dir, hdf5_normalize_dir, f"stats_jnt.h5")
    return 


def preprocess_audio_hdf5_normalize_state(cfg, dataset_name, data_pd, hdf5_dir, hdf5_normalize_dir):
    
    if dataset_name in ['VCC2020']: 
        preprocess_audio_hdf5_normalize_state_normal(cfg, data_pd, hdf5_dir, hdf5_normalize_dir)

    return 