import argparse
import pandas as pd
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
from Basic.config import hparams
from Basic.dataset import audio
from Basic.utils.folder_tools import *
from Basic.utils.hdf5_tools import *
from Basic.utils.train_tools import *
from VC.utils.cyclevae.infer_tools import VcCycleVaeInfer


def infer_reconst(args, cfg, hdf5_dir, csv_path, dataset_name):

    data_pd = pd.read_csv(csv_path)
    spk_list = list(set(data_pd['speaker'].tolist()))
    spk_list.sort()

    # init
    data_lists = []                 # {'dataset': [], 'speaker': [], 'section': [], 'utterance': [], 'key', [], 'file': [], 'state': [], 'mode': []}

    for spk_src in spk_list:

        # load cyclevae
        vc_cyclevae_infer = VcCycleVaeInfer(cfg)

        # load spk info
        vc_cyclevae_infer.load_spk_info(spk_src)
        
        # wav list 
        data_pd_spk = data_pd[data_pd['speaker'] == spk_src]
        wav_list = data_pd_spk['speaker'].tolist()
        wav_list.sort()

        for _, row in tqdm(data_pd_spk.iterrows(), "{}_{}".format(dataset_name, spk_src), total=len(wav_list)):  
            
            wav_path = row['file']
            key = row['key']
            key_name = key.split('.')[0]

            wav, wav_rec, feat_rec = vc_cyclevae_infer.voice_reconst(wav_path)

            # Save it on the disk
            # save hdf5
            output_dir = os.path.join(hdf5_dir, "world_reconst")
            create_folder(output_dir)

            state_path = os.path.join(output_dir, f"{key_name}.h5")
            write_hdf5(state_path, "wave", wav.astype(np.float32))
            write_hdf5(state_path, "wave_rec", wav_rec.astype(np.float32))
            write_hdf5(state_path, "feat_org_lf0", feat_rec)

            # save wav
            wav_reconst_dir = os.path.join(hdf5_dir, "wav_reconst")
            create_folder(wav_reconst_dir)
            audio.save_wav(wav_rec, os.path.join(wav_reconst_dir, key), cfg.dataset.sampling_rate)

            # data_lists
            data_lists.append({'dataset': row['dataset'], 'speaker': row['speaker'], 'section': row['section'], \
                                'utterance': row['utterance'], 'key': 'world_reconst_' + row['key'], 'file': row['file'], \
                                'state': state_path, 'mode': row['mode']})

    data_pd = pd.DataFrame(data_lists) 
    out_put_csv = os.path.join(os.path.dirname(csv_path), os.path.basename(str(csv_path)).split('_')[0] + '_reconst_' + '_'.join(os.path.basename(str(csv_path)).split('_')[1:]))
    data_pd.to_csv(out_put_csv, index=False, encoding="utf_8_sig")
    
    return  


def infer(args, mode_type):
    """
    模型推理
    """
    # load config
    cfg = load_cfg_file(args.config_file)

    # dataset
    for dataset_idx in range(len(cfg.general.dataset_list)):
        dataset_name = cfg.general.dataset_list[dataset_idx]

        print("Start infer reconst dataset: {}, mode_type: {}".format(dataset_name, mode_type))

        # init 
        hdf5_dir = os.path.join(cfg.general.data_dir, 'dataset_audio_hdf5', dataset_name)
        csv_path = os.path.join(cfg.general.data_dir, dataset_name + '_' + mode_type + '_hdf5.csv')

        # infer reconst
        infer_reconst(args, cfg, hdf5_dir, csv_path, dataset_name)
        print("Infer reconst dataset:{}  Done!".format(dataset_name))

    return 


def main(): 
    parser = argparse.ArgumentParser(description='Streamax VC Infer Engine')

    # # english
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_cyclevae.py", nargs='?', help='config file')

    # chinese
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VC/config/cyclevae/vc_config_chinese_cyclevae.py", nargs='?', help='config file')
    
    args = parser.parse_args()

    infer(args, hparams.TRAINING_NAME)
    infer(args, hparams.TESTING_NAME)


if __name__ == "__main__":
    main()