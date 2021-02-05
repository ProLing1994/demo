import argparse
import sys
import time
from tqdm import tqdm

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/KWS')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/KWS')
from utils.train_tools import *

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
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
        score = net(x)
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


def train(config_file, local_rank, training_mode):
    """ training engine
    :param config_file:   the input configuration file
    :param local_rank:    the device index
    :param training_mode: the model training mode
    :return:              None
    """
    # record time
    begin_t = time.time()

    # load configuration file
    cfg = load_cfg_file(config_file)

    # check 
    assert cfg.general.data_parallel_mode == 1, "[ERROR] If you want DataParallel, please run train.py"

    # clean the existing folder if the user want to train from scratch
    setup_workshop(cfg)

    # control randomness during training
    init_torch_and_numpy(cfg, local_rank)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
    logger = setup_logger(log_file, 'kws_train')

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # define loss function
    loss_func = define_loss_function(cfg)

    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)

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
        start_epoch = last_save_epochyes
    else:
        start_epoch, last_save_epoch, start_batch = 0, 0, 0

    # get training data set and test data set
    train_dataloader, train_sampler, len_dataset = generate_dataset_ddp(
        cfg, 'training', training_mode)
    if cfg.general.is_test:
        eval_validation_dataloader = generate_test_dataset(
            cfg, 'validation', training_mode=training_mode)

    msg = 'Training dataset number: {}'.format(len_dataset)
    logger.info(msg)

    msg = 'Init Time: {}'.format((time.time() - begin_t) * 1.0)
    logger.info(msg)

    # loop over batches
    for epoch_idx in range(cfg.train.num_epochs - (cfg.general.resume_epoch if cfg.general.resume_epoch != -1 else 0)):

        train_sampler.set_epoch(epoch_idx)

        for batch_idx, (inputs, labels, indexs) in enumerate(train_dataloader):

            net.train()
            begin_t = time.time()
            optimizer.zero_grad()

            # save training images for visualization
            if cfg.debug.save_inputs:
                save_intermediate_results(cfg, "training", epoch_idx, inputs, labels, indexs)

            inputs, labels = inputs.cuda(), labels.cuda()
            scores = net(inputs)
            loss = loss_func(scores, labels)
            loss.backward()
            optimizer.step()

            # 仅仅只有主进程打印日志，保存模型
            if local_rank == 0:

                # caltulate accuracy
                pred_y = torch.max(scores, 1)[1].cpu().data.numpy()
                accuracy = float((pred_y == labels.cpu().data.numpy()).astype(
                    int).sum()) / float(labels.size(0))

                # print training information
                sample_duration = (time.time() - begin_t) * \
                    1.0 / cfg.train.batch_size
                
                epoch_num = start_epoch + epoch_idx
                batch_num = epoch_num * len_dataset // cfg.train.batch_size + batch_idx
                msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_loss: {:.4f}, time: {:.4f} s/vol' \
                    .format(epoch_num, batch_num, accuracy, loss.item(), sample_duration)
                logger.info(msg)

                if (batch_num % cfg.train.plot_snapshot) == 0:
                    plot_tool(cfg, log_file)

                if epoch_num % cfg.train.save_epochs == 0 or epoch_num == cfg.train.num_epochs - 1:
                    if last_save_epoch != epoch_num:
                        last_save_epoch = epoch_num

                        # save training model
                        save_checkpoint(net, optimizer, epoch_num,
                                        batch_num, cfg, config_file)

                        if cfg.general.is_test:
                            test(cfg, net, loss_func, epoch_num, batch_num, logger, eval_validation_dataloader, mode='eval')
                            # test(cfg, net, loss_func, epoch_num, batch_num, logger, eval_train_dataloader, mode='eval')


def main():
    parser = argparse.ArgumentParser(
        description='Streamax KWS Training Engine')
    parser.add_argument('-i', '--input', type=str,
                        default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # training mode: [0,1]
    # 0: 将整个唤醒词作为一个 label 进行建模
    # 1: 根据帧对齐结果，采用更简洁的建模方式，对转音位置进行建模，一个唤醒词拥有多个标签
    training_mode = 0
    if 'align' in args.input:
        training_mode = 1

    train(args.input, args.local_rank, training_mode)


if __name__ == "__main__":
    main()
