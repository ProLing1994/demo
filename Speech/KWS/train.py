import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.utils.loss_tools import *
from Basic.utils.profiler_tools import Profiler
from Basic.utils.train_tools import *

from KWS.utils.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/yuanhuan/code/demo/common')
from common.utils.python.logging_helpers import setup_logger


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
    for _, (input, label, index) in tqdm(enumerate(test_data_loader)):
        input, label = input.cuda(), label.cuda()

        if cfg.dataset.h_alignment == True:
            hisi_input = input[:, :, :(input.shape[2] // 16) * 16, :]
            if cfg.loss.method == 'classification':
                score = net(hisi_input)
            elif cfg.loss.method == 'embedding':
                _ = net(hisi_input)
            elif cfg.loss.method == 'classification & embedding':
                _, score = net(hisi_input)
            else:
                raise Exception("[Unknow:] cfg.loss.method. ")
        else:
            if cfg.loss.method == 'classification':
                score = net(input)
            elif cfg.loss.method == 'embedding':
                _ = net(input)
            elif cfg.loss.method == 'classification & embedding':
                _, score = net(input)
            else:
                raise Exception("[Unknow:] cfg.loss.method. ")
        score = score.view(score.size()[0], score.size()[1])
        loss = loss_func(score, label)

        scores.append(torch.max(score, 1)[1].cpu().data.numpy())
        labels.append(label.cpu().data.numpy())
        losses.append(loss.item())

    # Calculate accuracy
    accuracy = float((np.array(scores) == np.array(labels)
                      ).astype(int).sum()) / float(len(labels))
    
    # Calculate loss
    loss = np.array(losses).sum()/float(len(labels))

    # Show information
    msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_loss: {:.4f}'.format(
        epoch_idx, batch_idx, mode, accuracy, mode, loss)
    logger.info(msg)


def train(config_file, training_mode):
    """ training engine
    :param config_file:   the input configuration file
    :param training_mode: the model training mode
    :return:              None
    """
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
    loss_func = loss_function(cfg)
    if cfg.loss.method == 'embedding' or cfg.loss.method == 'classification & embedding':
        loss_func_embedding = loss_function_embedding(cfg)

    # ema
    if cfg.loss.ema_on:
        ema = EMA(net, 0.9999)

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        load_checkpoint(net, 
                        cfg.general.load_mode_type,
                        cfg.general.finetune_model_dir, cfg.general.finetune_epoch_num, cfg.general.finetune_sub_folder_name,
                        cfg.general.finetune_model_path,
                        cfg.general.finetune_state_name, cfg.general.finetune_ignore_key_list, cfg.general.finetune_add_module_type)

        start_epoch, start_batch = 0, 0
        last_save_epoch = 0
    elif cfg.general.resume_epoch_num >= 0:
        # resume, Load the model, continue the previous learning rate
        start_epoch, start_batch = load_checkpoint(net, 
                                    cfg.general.load_mode_type,
                                    cfg.general.save_dir, cfg.general.resume_epoch_num, cfg.general.finetune_sub_folder_name,
                                    cfg.general.finetune_model_path,
                                    cfg.general.finetune_state_name, cfg.general.finetune_ignore_key_list, cfg.general.finetune_add_module_type, 
                                    optimizer=optimizer)
        last_save_epoch = start_epoch
    else:
        start_epoch, start_batch = 0, 0
        last_save_epoch = 0

    # knowledge distillation
    if cfg.knowledge_distillation.on:
        msg = 'Knowledge Distillation: {} -> {}'.format(cfg.knowledge_distillation.teacher_model_name, cfg.net.model_name)
        logger.info(msg)

        teacher_model = import_network(cfg, cfg.knowledge_distillation.teacher_model_name, 
                                        cfg.knowledge_distillation.teacher_class_name)
        load_checkpoint(teacher_model, 
                        0,
                        cfg.knowledge_distillation.teacher_model_dir, cfg.knowledge_distillation.epoch, 'checkpoints',
                        "",
                        'state_dict', [], 0)
        teacher_model.eval()

    # define training dataset and testing dataset
    train_dataloader, len_train_dataset = generate_dataset(cfg, 'training', training_mode)
    if cfg.general.is_test:
        eval_validation_dataloader = generate_test_dataset(cfg, 'validation', training_mode=training_mode)
        # eval_train_dataloader = generate_test_dataset(cfg, 'training')

    msg = 'Training dataset number: {}'.format(len_train_dataset)
    logger.info(msg)

    batch_number = len(train_dataloader)
    data_iter = iter(train_dataloader)
    batch_idx = start_batch

    # profiler
    profiler = Profiler(summarize_every=cfg.train.show_log, disabled=False)

    # loop over batches
    for i in range(batch_number):

        net.train()

        epoch_idx = start_epoch + i * cfg.train.batch_size // len_train_dataset
        batch_idx += 1

        # Blocking, waiting for batch (threaded)
        inputs, labels, indexs = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # save training images for visualization
        if cfg.debug.save_inputs:
            save_intermediate_results(cfg, "training", epoch_idx, inputs, labels, indexs)
        
        # Data to device
        inputs, labels = inputs.cuda(), labels.cuda()
        profiler.tick("Data to device")

        # Forward pass
        # h_alignment，模型需要图像输入长度为 16 的倍数
        # 此处进行该操作的原因，便于后面使用知识蒸馏，teacher 模型可以使用原数据作为输入
        if cfg.dataset.h_alignment == True:
            hisi_input = inputs[:, :, :(inputs.shape[2] // 16) * 16, :]
            if cfg.loss.method == 'classification':
                scores = net(hisi_input)
            elif cfg.loss.method == 'embedding':
                embeds = net(hisi_input)
            elif cfg.loss.method == 'classification & embedding':
                embeds, scores = net(hisi_input)
            else:
                raise Exception("[Unknow:] cfg.loss.method. ")
        else:
            if cfg.loss.method == 'classification':
                scores = net(inputs)
            elif cfg.loss.method == 'embedding':
                embeds = net(inputs)
            elif cfg.loss.method == 'classification & embedding':
                embeds, scores = net(inputs)
            else:
                raise Exception("[Unknow:] cfg.loss.method. ")
        scores = scores.view(scores.size()[0], scores.size()[1])
        if cfg.loss.method == 'embedding' or cfg.loss.method == 'classification & embedding':
            embeds = embeds.view(embeds.size()[0], embeds.size()[1])
        profiler.tick("Forward pass")
        
        # Calculate loss
        if cfg.loss.method == 'classification':
            loss = loss_func(scores, labels)
        elif cfg.loss.method == 'embedding':
            loss = loss_func_embedding(embeds, labels)
        elif cfg.loss.method == 'classification & embedding':
            loss = loss_func(scores, labels) + cfg.loss.embedding_weight * loss_func_embedding(embeds, labels)
        if cfg.knowledge_distillation.on:
            teacher_model.eval()
            teacher_scores = teacher_model(inputs)
            loss = loss_fn_kd(cfg, scores, teacher_scores, loss)
        profiler.tick("Calculate Loss")

        # Backward pass
        net.zero_grad()
        optimizer.zero_grad()
        loss.backward()

        profiler.tick("Backward pass")

        # Parameter update
        optimizer.step()
        update_scheduler(cfg, scheduler, epoch_idx)
        profiler.tick("Parameter update")

        if cfg.loss.ema_on:
            ema.update_params()     # apply ema

        # Caltulate accuracy
        pred_y = torch.max(scores, 1)[1].cpu().data.numpy()
        accuracy = float((pred_y == labels.cpu().data.numpy()).astype(
            int).sum()) / float(labels.size(0))
        profiler.tick("Caltulate accuracy")

        # Show information
        if (batch_idx % cfg.train.show_log) == 0:
            msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_loss: {:.4f}' \
                .format(epoch_idx, batch_idx, accuracy, loss.item())
            logger.info(msg)
        profiler.tick("Show information")

        # Plot snapshot
        if (batch_idx % cfg.train.plot_snapshot) == 0:
            plot_tool(cfg, log_file)
        profiler.tick("Plot snapshot")

        # Save model
        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                if cfg.loss.ema_on:
                    ema.apply_shadow() # copy ema status to the model

                # save training model
                save_checkpoint(cfg, config_file, net, optimizer, epoch_idx, batch_idx)

                if cfg.general.is_test:
                    test(cfg, net, loss_func, epoch_idx, batch_idx,
                         logger, eval_validation_dataloader, mode='eval')
            
                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")


def main():
    parser = argparse.ArgumentParser(description='Streamax KWS Training Engine')

    # training_mode = 0
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_speech.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoyu.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaole.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui8k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaorui16k.py", nargs='?', help='config file')
    # parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_xiaoan8k.py", nargs='?', help='config file')
    parser.add_argument('-i', '--input', type=str, default="/home/huanyuan/code/demo/Speech/KWS/config/kws/kws_config_embedding_xiaoan8k.py", nargs='?', help='config file')
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
