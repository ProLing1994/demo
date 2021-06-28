import argparse
import sys
import time

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech/SED')
sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SED')
from utils.folder_tools import *
from utils.train_tools import *
from utils.optimizer_tools import *
from utils.loss_tools import *

# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo')
# sys.path.insert(0, '/yuanhuan/code/demo')
sys.path.insert(0, '/home/huanyuan/code/demo')
from common.common.utils.python.logging_helpers import setup_logger

def test(cfg, net, loss_func, epoch_idx, iteration, logger, test_data_loader, mode='eval'):
    """
    :param cfg:                config contain data set information
    :param net:                net
    :param epoch_idx:          epoch index
    :param iteration:          iteration index
    :param logger:             log for save testing result
    :param test_data_loader:   testing data loader
    :param mode:               evaluate either training set or testing set
    :return:                   None
    """
    scores = []
    labels = []
    losses = []
    for _, (x, label, index) in tqdm(enumerate(test_data_loader)):
        x, label = x.cuda(), label.cuda()
        
        net.eval()
        score = net(x)
        score = score.view(score.size()[0], score.size()[1])
        loss = loss_func(score, label)

        score, label = calculate_score_label(cfg, score, label)
        scores.extend(score)
        labels.extend(label)
        losses.append(loss.item())

    loss = np.array(losses).sum() / float(len(losses))
    accuracy = calculate_accuracy(cfg, np.array(scores), np.array(labels))
    # caltulate accuracy
    if cfg.dataset.label.type == "multi_class":
        msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_loss: {:.4f}'.format(epoch_idx, iteration, mode, accuracy, mode, loss)
    elif cfg.dataset.label.type == "multi_label":
        mAP, mAUC, dprime = calculate_mAP_mAUC_dprime(np.array(scores), np.array(labels))
        msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_mAP: {:.4f}, {}_mAUC: {:.4f}, {}_dprime: {:.4f}, {}_loss: {:.4f}' \
            .format(epoch_idx, iteration, mode, accuracy, mode, mAP, mode, mAUC, mode, dprime, mode, loss)

    logger.info(msg)


def train(args):
    # record time
    begin_t = time.time()

    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # check
    assert cfg.general.data_parallel_mode == 0, "[ERROR] If you want DistributedDataParallel, please run train_dpp.py"

    # clean the existing folder if the user want to train from scratch
    setup_workshop(cfg)
    
    # control randomness during training
    init_torch_and_numpy(cfg)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
    logger = setup_logger(log_file, 'sed_train')

    # model
    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # set training optimizer, learning rate scheduler
    optimizer = Optimizer(cfg, net)

    # define loss function
    loss_func = define_loss_function(cfg)

    # get training data set and test data set
    train_dataloader, train_datasampler, len_dataset = generate_dataset(cfg, 'training')
    if cfg.general.is_test:
        eval_validation_dataloader = generate_test_dataset(cfg, 'testing')

    # iteration
    iteration = 0

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        load_checkpoint(net, 
                        cfg.general.finetune_epoch,
                        cfg.general.finetune_model_dir, 
                        sub_folder_name='pretrain_model')
    elif cfg.general.resume_epoch >= 0:
        # resume, Load the model, continue the previous learning rate, sampler
        iteration = load_checkpoint(net,
                        cfg.general.resume_epoch, 
                        cfg.general.save_dir, 
                        optimizer=optimizer,
                        sampler=train_datasampler)

    # last_save_epoch
    last_save_epoch = iteration * cfg.train.batch_size // len_dataset
    # last_show_epoch
    last_show_epoch = iteration * cfg.train.batch_size // len_dataset

    msg = 'Training dataset number: {}'.format(len_dataset)
    logger.info(msg)

    msg = 'Init Time: {}'.format((time.time() - begin_t) * 1.0)
    logger.info(msg)
    
    # list init
    scores_list = []
    labels_list = []
    losses_list = []
    for _, (inputs, labels, indexs) in enumerate(train_dataloader): 
        epoch_idx = iteration * cfg.train.batch_size // len_dataset
        
        # Save Model and test
        if epoch_idx % cfg.train.save_epochs == 0 and epoch_idx != last_save_epoch:
            last_save_epoch = epoch_idx

            # save model
            msg = 'Save Model: {}'.format(epoch_idx)
            logger.info(msg)
            save_checkpoint(cfg, args.config_file, net, optimizer, train_datasampler, epoch_idx, iteration)

            # test model
            msg = 'Test Model: {}'.format(epoch_idx)
            logger.info(msg)
            if cfg.general.is_test:
                test(cfg, net, loss_func, epoch_idx, iteration, logger, eval_validation_dataloader, mode='eval')

        # Stop learning
        if epoch_idx == cfg.train.num_epochs:
            msg = 'Train Done. '
            logger.info(msg)        
            
            # visualization
            plot_tool(cfg, log_file)
            break

        inputs, labels = inputs.cuda(), labels.cuda()
        # print(labels.cpu().data.sort())

        # mix up
        bool_mix_up = np.random.uniform(0, 1) < cfg.dataset.augmentation.mix_up_frequency
        if cfg.dataset.augmentation.on and cfg.dataset.augmentation.mix_up_on and bool_mix_up:
            net.train()
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, cfg.dataset.augmentation.mix_up_alpha)
            scores = net(inputs)
            scores = scores.view(scores.size()[0], scores.size()[1])
            loss = mixup_criterion(loss_func, scores, targets_a, targets_b, lam)
        else:
            net.train()
            scores = net(inputs)
            scores = scores.view(scores.size()[0], scores.size()[1])
            loss = loss_func(scores, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # caltulate accuracy
        if cfg.dataset.augmentation.mix_up_on and bool_mix_up:
            if lam >= 0.5:
                scores, labels = calculate_score_label(cfg, scores, targets_a)
            else:
                scores, labels = calculate_score_label(cfg, scores, targets_b)
        else:
            scores, labels = calculate_score_label(cfg, scores, labels)

        scores_list.extend(scores)
        labels_list.extend(labels)
        losses_list.append(loss.item())

        # information
        print('epoch: {}, batch: {}'.format(epoch_idx, iteration), end='\r')
        if epoch_idx != last_show_epoch:
            last_show_epoch = epoch_idx
    
            loss = np.array(losses_list).sum() / float(len(losses_list))
            accuracy = calculate_accuracy(cfg, np.array(scores_list), np.array(labels_list))
            msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_loss: {:.4f}' \
                .format(epoch_idx, iteration, accuracy, loss)
            logger.info(msg)

            # visualization
            plot_tool(cfg, log_file)
            
            # list init
            scores_list = []
            labels_list = []
            losses_list = []

        # save images for visualization
        if cfg.debug.save_inputs:
            save_intermediate_results(cfg, "training", epoch_idx, inputs.cpu(), labels.cpu(), indexs)

        iteration += 1

if __name__ == '__main__':
    """
    功能描述：模型训练和测试脚本
    """
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')
    # parser.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_FSD50K.py')
    args = parser.parse_args()

    train(args)

    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers()
    # parser_create_indexes = subparsers.add_parser('train')
    # parser_create_indexes.add_argument('-i', '--config_file', type=str, default='/home/huanyuan/code/demo/Speech/SED/config/sed_config_ESC50.py')
    # parser_create_indexes.set_defaults(func=train)   
    # args = parser.parse_args()

    # args.func(args)