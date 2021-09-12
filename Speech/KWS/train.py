import argparse
import sys
import time
from tqdm import tqdm

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech/KWS')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *
from utils.loss_tools import *

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.logging_helpers import setup_logger

def test(cfg, net, loss_func, epoch_idx, batch_idx, logger, test_data_loader, mode='eval'):
    """
    :param cfg:                config contain data set information
    :param net:                net
    :param epoch_idx:          epoch index
    :param batch_idx:          batch index
    :param logger:             log for save testing result
    :param test_data_loader:   testing data loader
    :param mode:               evaluate either training set or testing set
    :return:                   None
    """
    net.eval()

    scores = []
    labels = []
    losses = []
    for _, (x, label, index) in tqdm(enumerate(test_data_loader)):
        x, label = x.cuda(), label.cuda()

        if cfg.dataset.h_alignment == True:
            hisi_input = x[:, :, :(x.shape[2] // 16) * 16, :]
            score = net(hisi_input)
        else:
            score = net(x)
        score = score.view(score.size()[0], score.size()[1])
        loss = loss_func(score, label)

        scores.append(torch.max(score, 1)[1].cpu().data.numpy())
        labels.append(label.cpu().data.numpy())
        losses.append(loss.item())

    # caltulate accuracy
    accuracy = float((np.array(scores) == np.array(labels)
                      ).astype(int).sum()) / float(len(labels))
    loss = np.array(losses).sum()/float(len(labels))

    msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_loss: {:.4f}'.format(
        epoch_idx, batch_idx, mode, accuracy, mode, loss)
    logger.info(msg)


def train(config_file, training_mode):
    """ training engine
    :param config_file:   the input configuration file
    :param training_mode: the model training mode
    :return:              None
    """
    # record time
    begin_t = time.time()

    # load configuration file
    cfg = load_cfg_file(config_file)

    # check
    assert cfg.general.data_parallel_mode == 0, "[ERROR] If you want DistributedDataParallel, please run train_dpp.py"
    assert not cfg.deep_mutual_learning.on, "[ERROR] If you  want Deep Mutual Learning, please run train_dml.py"

    # clean the existing folder if the user want to train from scratch
    setup_workshop(cfg)

    # control randomness during training
    init_torch_and_numpy(cfg)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
    logger = setup_logger(log_file, 'kws_train')

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)
    scheduler = set_scheduler(cfg, optimizer)

    # define loss function
    loss_func = define_loss_function(cfg)

    # ema
    if cfg.loss.ema_on:
        ema = EMA(net, 0.9999)

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        last_save_epoch, start_batch = load_checkpoint(cfg.general.finetune_epoch, net,
                                                        cfg.general.finetune_model_dir, 
                                                        sub_folder_name='pretrain_model')
        start_epoch, last_save_epoch, start_batch = 0, 0, 0
    elif cfg.general.resume_epoch >= 0:
        # resume, Load the model, continue the previous learning rate
        last_save_epoch, start_batch = load_checkpoint(cfg.general.resume_epoch, net,
                                                        cfg.general.save_dir, 
                                                        optimizer=optimizer)
        start_epoch = last_save_epoch
    else:
        start_epoch, last_save_epoch, start_batch = 0, 0, 0
        # start_epoch, last_save_epoch, start_batch = 0, -1, 0

    # knowledge distillation
    if cfg.knowledge_distillation.on:
        msg = 'Knowledge Distillation: {} -> {}'.format(cfg.knowledge_distillation.teacher_model_name, cfg.net.model_name)
        logger.info(msg)

        teacher_model = import_network(cfg, model_name=cfg.knowledge_distillation.teacher_model_name)
        _, _ = load_checkpoint(cfg.knowledge_distillation.epoch, teacher_model,
                                    cfg.knowledge_distillation.teacher_model_dir)
        teacher_model.eval()

    # define training dataset and testing dataset
    train_dataloader, len_dataset = generate_dataset(cfg, 'training', training_mode)
    if cfg.general.is_test:
        eval_validation_dataloader = generate_test_dataset(cfg, 'validation', training_mode=training_mode)
        # eval_train_dataloader = generate_test_dataset(cfg, 'training')

    msg = 'Training dataset number: {}'.format(len_dataset)
    logger.info(msg)

    msg = 'Init Time: {}'.format((time.time() - begin_t) * 1.0)
    logger.info(msg)

    batch_number = len(train_dataloader)
    data_iter = iter(train_dataloader)
    batch_idx = start_batch

    # # save model 
    # os.makedirs('/mnt/huanyuan/model/kws_xiaorui_12162020_test/checkpoints/chk_1/')
    # torch.save(net.cpu().module.state_dict(), '/mnt/huanyuan/model/kws_xiaorui_12162020_test/checkpoints/chk_1/net_parameter.pth')

    # loop over batches
    for i in range(batch_number):

        net.train()
        begin_t = time.time()
        optimizer.zero_grad()

        epoch_idx = start_epoch + i * cfg.train.batch_size // len_dataset
        batch_idx += 1

        inputs, labels, indexs = data_iter.next()

        # save training images for visualization
        if cfg.debug.save_inputs:
            save_intermediate_results(cfg, "training", epoch_idx, inputs, labels, indexs)

        # Forward pass
        inputs, labels = inputs.cuda(), labels.cuda()
        if cfg.dataset.h_alignment == True:
            hisi_input = inputs[:, :, :(inputs.shape[2] // 16) * 16, :]
            scores = net(hisi_input)
        else:
            scores = net(inputs)
        scores = scores.view(scores.size()[0], scores.size()[1])
        
        # Calculate loss
        loss = loss_func(scores, labels)
        if cfg.knowledge_distillation.on:
            teacher_model.eval()
            teacher_scores = teacher_model(inputs)
            loss = loss_fn_kd(cfg, scores, teacher_scores, loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        update_scheduler(cfg, scheduler, epoch_idx)

        if cfg.loss.ema_on:
            ema.update_params()     # apply ema

        # caltulate accuracy
        pred_y = torch.max(scores, 1)[1].cpu().data.numpy()
        accuracy = float((pred_y == labels.cpu().data.numpy()).astype(
            int).sum()) / float(labels.size(0))

        # print training information
        sample_duration = (time.time() - begin_t) * 1.0 / cfg.train.batch_size
        msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_loss: {:.4f}, time: {:.4f} s/vol' \
            .format(epoch_idx, batch_idx, accuracy, loss.item(), sample_duration)
        logger.info(msg)

        if (batch_idx % cfg.train.plot_snapshot) == 0:
            plot_tool(cfg, log_file)

        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
        # if epoch_idx == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                if cfg.loss.ema_on:
                    ema.apply_shadow() # copy ema status to the model

                # save training model
                save_checkpoint(cfg, config_file, net, optimizer, epoch_idx, batch_idx)

                if cfg.general.is_test:
                    test(cfg, net, loss_func, epoch_idx, batch_idx,
                         logger, eval_validation_dataloader, mode='eval')
                    # test(cfg, net, loss_func, epoch_idx, batch_idx, logger, eval_train_dataloader, mode='eval')
            
                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters


def main():
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')

    # training_mode = 0
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_speech.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui8k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui16k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan8k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan16k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_pretrain.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_all_pretrain.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_2_label_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_activatebwc.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_heybodycam.py", nargs='?', help='config file')

    # training_mode = 1
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_xiaorui.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_align_pretrain.py", nargs='?', help='config file')
    args = parser.parse_args()

    # training mode: [0,1]
    # 0: 将整个唤醒词作为一个 label 进行建模
    # 1: 根据帧对齐结果，采用更简洁的建模方式，对转音位置进行建模，一个唤醒词拥有多个标签
    training_mode = 0
    if 'align' in args.input:
        training_mode = 1

    train(args.input, training_mode)


if __name__ == "__main__":
    main()
