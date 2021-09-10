import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from utils.train_tools import *
from utils.loss_tools import *
from config.hparams import *

sys.path.insert(0, '/home/huanyuan/code/demo/common/common')
from utils.python.logging_helpers import setup_logger


def train(args):
    """ training engine
    :param config_file:   the input configuration file
    :return:              None
    """
    # load configuration file
    cfg = load_cfg_file(args.config_file)

    # clean the existing folder if the user want to train from scratch
    setup_workshop(cfg)

    # control randomness during training
    init_torch_and_numpy(cfg)

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
    logger = setup_logger(log_file, 'sv_train')

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)
    scheduler = set_scheduler(cfg, optimizer)

    # define loss function
    loss_func = loss_function(cfg)

    # ema
    if cfg.loss.ema_on:
        ema = EMA(net, 0.9999)

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        last_save_epoch, start_batch = load_checkpoint(net, cfg.general.finetune_epoch, 
                                                        cfg.general.finetune_model_dir, 
                                                        sub_folder_name='pretrain_model')
        start_epoch, last_save_epoch, start_batch = 0, 0, 0
    if cfg.general.resume_epoch >= 0:
        # resume, Load the model, continue the previous learning rate
        last_save_epoch, start_batch = load_checkpoint(net, cfg.general.resume_epoch,
                                                        cfg.general.save_dir, 
                                                        optimizer=optimizer)
        start_epoch = last_save_epoch
    else:
        start_epoch, last_save_epoch, start_batch = 0, 0, 0

    # knowledge distillation
    if cfg.knowledge_distillation.on:
        msg = 'Knowledge Distillation: {} -> {}'.format(cfg.knowledge_distillation.teacher_model_name, cfg.net.model_name)
        logger.info(msg)

        teacher_model = import_network(cfg, model_name=cfg.knowledge_distillation.teacher_model_name)
        _, _ = load_checkpoint(teacher_model, cfg.knowledge_distillation.epoch, 
                                cfg.knowledge_distillation.teacher_model_dir)
        teacher_model.eval()

    # define training dataset and testing dataset
    train_dataloader, len_train_dataset = generate_dataset(cfg, TRAINING_NAME)

    msg = 'Training dataset number: {}'.format(len_train_dataset)
    logger.info(msg)

    batch_number = len(train_dataloader)
    data_iter = iter(train_dataloader)
    batch_idx = start_batch

    # loop over batches
    for i in range(batch_number):

        net.train()
        optimizer.zero_grad()

        epoch_idx = start_epoch + i * cfg.train.batch_size // len_train_dataset
        batch_idx += 1

        data = data_iter.next()
        
        print(data.shape)

def main(): 
    parser = argparse.ArgumentParser(description='Streamax SV Training Engine')
    args = parser.parse_args()
    args.config_file = "/home/huanyuan/code/demo/Speech/SV/config/sv_config_TI_SV.py"
    train(args)


if __name__ == "__main__":
    main()