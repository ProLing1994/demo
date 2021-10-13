import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.profiler_tools import *
from Basic.utils.loss_tools import *
from Basic.utils.train_tools import *

from SV.utils.train_tools import *
from SV.utils.infer_tools import *
from SV.utils.loss_tools import *
from SV.utils.visualizations_tools import *
from SV.dataset.sv_dataset_preload_audio_lmdb import *


def infer(args):
    """
    模型推理，通过滑窗的方式得到每一小段 embedding，随后计算 EER
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # load prediction model
    load_checkpoint(net, cfg.test.model_epoch, cfg.general.save_dir)
    net.eval()

    # load data
    data_pd = load_data_pd(cfg, hparams.TESTING_NAME)
    lmdb_dict = load_lmdb(cfg, hparams.TESTING_NAME)
    
    embeds_dict = {}
    for idx, row in tqdm(data_pd.iterrows(), total=len(data_pd)):
        # init
        data = read_audio_lmdb(lmdb_dict[str(row['dataset'])], str(row['file']))
        assert len(data) > 0, "{} {}".format(str(row['dataset']), str(row['file']))
        tqdm.write('{}: shape: {}, path :{}'.format(idx, data.shape, str(row['file'])))
        speaker = str(row['speaker'])

        # embed
        embed = embedding(cfg, net, data)

        if speaker in embeds_dict:
            embeds_dict[speaker].append(embed)
        else:
            embeds_dict[speaker] = []
            embeds_dict[speaker].append(embed)

    embeds_list = []
    for speaker_name in embeds_dict.keys():
        embedding_per_speaker_list = []

        for embed_idx in range(len(embeds_dict[speaker_name])):
            embedding_per_speaker_list.append(embeds_dict[speaker_name][embed_idx])
        
        embeds_list.append(embedding_per_speaker_list)

    # Calculate loss
    embedding_lens = [len(embedding_per_speaker_list) for embedding_per_speaker_list in embeds_list]
    min_embedding_len = min(embedding_lens)
    print("min_embedding_len: {}".format(min_embedding_len))

    embeds_list = [embedding_per_speaker_list[: min_embedding_len] for embedding_per_speaker_list in embeds_list]
    embeds_np = np.array(embeds_list)

    if isinstance(net, torch.nn.parallel.DataParallel):
        sim_matrix = net.module.similarity_matrix_cpu(embeds_np)
    else:
        sim_matrix = net.similarity_matrix_cpu(embeds_np)
    eer = compute_eer(embeds_np, sim_matrix)
    print("eer: {:.4f}".format(eer))

    projection_fpath = os.path.join(cfg.general.save_dir, "umap_infer_epoch_{}_eer_{:.4f}.png".format(cfg.test.model_epoch, eer))
    draw_projections(embeds_np, cfg.test.model_epoch, projection_fpath)


def main():
    parser = argparse.ArgumentParser(description='Streamax SV Training Engine')
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", nargs='?', help='config file')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", nargs='?', help='config file')
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()