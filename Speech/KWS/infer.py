import argparse
import pandas as pd
import sys
import torch.nn.functional as F

from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from impl.pred_pyimpl import kws_load_model
from dataset.kws.dataset_helper import load_label_index
from utils.train_tools import *


def predict(config_file, epoch, mode, augmentation_on):
    # load configuration file
    cfg = load_cfg_file(config_file)

    # load prediction model
    model = kws_load_model(cfg.general.save_dir, int(cfg.general.gpu_ids), epoch)
    net = model['prediction']['net']
    net.eval()

    # load label index 
    label_index = load_label_index(cfg.dataset.label.positive_label, cfg.dataset.label.negative_label)

    # load data 
    data_pd = pd.read_csv(cfg.general.data_csv_path)
    data_pd_mode = data_pd[data_pd['mode'] == mode]

    # test dataset
    test_data_loader = generate_test_dataset(cfg, mode, augmentation_on=augmentation_on)
    # test_data_loader = generate_test_dataset(cfg, mode, augmentation_on=True)

    results_list = []
    preds = []
    labels = []
    for _, (x, label, index) in tqdm(enumerate(test_data_loader)):
        results_dict = {}

        results_dict['file'] = data_pd_mode['file'].tolist()[index]
        results_dict['mode'] = data_pd_mode['mode'].tolist()[index]
        results_dict['label'] = data_pd_mode['label'].tolist()[index]
        assert results_dict['mode']  == mode, "[ERROR:] Something wronge about mode, please check"

        # gen label
        results_dict['label_idx'] = label_index[results_dict['label']]
        assert results_dict['label_idx']  == label, "[ERROR:] Something wronge about mode, please check"

        x, label = x.cuda(), label.cuda()
        score = net(x)
        score = F.softmax(score, dim=1)
        
        pred = torch.max(score, 1)[1].cpu().data.numpy()
        preds.append(pred)
        labels.append(label.cpu().data.numpy())

        results_dict['result_idx'] = pred[0]
        score = score.cpu().data.numpy()
        for label_idx in range(score.shape[1]):
            results_dict['prob_{}'.format(label_idx)] = score[0][label_idx]
        
        results_list.append(results_dict)

    # caltulate accuracy
    accuracy = float((np.array(preds) == np.array(labels)).astype(int).sum()) / float(len(labels))
    msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}'.format(model['prediction']['epoch'], model['prediction']['batch'], mode, accuracy)
    print(msg)

    # out csv
    csv_data_pd = pd.DataFrame(results_list)
    csv_data_pd.to_csv(os.path.join(cfg.general.save_dir, 'infer_{}_augmentation_{}.csv'.format(mode, augmentation_on)), index=False, encoding="utf_8_sig")


def main():
    """
    使用模型对音频文件进行测试，配置为 --input 中的 config 文件，当存在音频文件长度大于模型送入的音频文件长度时(1s\2s), 该脚本会随机截取一段音频进行测试，将测试结果作为整段音频的测试结果，
    在训练过程中，随机截取可以作为一种数据增强手段(需要考虑是否出现问题)，在测试过程中，随机截取存在误差，同时每次结果不同
    """
    # default_mode = "testing,validation,training"
    default_mode = "testing,validation"
    default_model_epoch = -1
    default_augmentation_on = False

    parser = argparse.ArgumentParser(description='Streamax KWS Infering Engine')
    # parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config.py", help='config file')
    parser.add_argument('--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", help='config file')
    parser.add_argument('--mode', type=str, default=default_mode)
    parser.add_argument('--epoch', type=str, default=default_model_epoch)
    parser.add_argument('--augmentation_on', type=bool, default=default_augmentation_on)
    args = parser.parse_args()

    mode_list = args.mode.strip().split(',')
    for mode_type in mode_list:
        predict(args.input, args.epoch, mode_type, args.augmentation_on)


if __name__ == "__main__":
    main()
