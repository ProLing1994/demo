import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech/SV')
from utils.folder_tools import *
from utils.train_tools import *
from utils.loss_tools import *
from utils.profiler_tools import *
from utils.visualizations_tools import *
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

    # profiler
    profiler = Profiler(summarize_every=cfg.train.show_log, disabled=False)

    # loop over batches
    for i in range(batch_number):

        net.train()

        epoch_idx = start_epoch + i * cfg.train.batch_size // len_train_dataset
        batch_idx += 1

        # Blocking, waiting for batch (threaded)
        inputs = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Data to device
        inputs = torch.from_numpy(inputs).float().cuda()
        profiler.tick("Data to device")
        
        # Forward pass
        if cfg.dataset.h_alignment == True:
            hisi_input = inputs[:, :, :(inputs.shape[2] // 16) * 16, :]
            embeds, sim_matrix = net(hisi_input)
        else:
            embeds, sim_matrix = net(inputs)
        profiler.tick("Forward pass")

        # Calculate loss
        loss, eer = ge2e_loss(embeds, sim_matrix, loss_func)

        if cfg.knowledge_distillation.on:
            teacher_model.eval()
            teacher_embeds, _ = teacher_model(inputs)
            # TO
            pass
        profiler.tick("Calculate Loss")
        
        # Backward pass
        net.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        profiler.tick("Backward pass")

        # Parameter update
        if isinstance(net, torch.nn.parallel.DataParallel):
            net.module.do_gradient_ops()
        else:
            net.do_gradient_ops()
        
        optimizer.step()
        update_scheduler(cfg, scheduler, epoch_idx)
        profiler.tick("Parameter update")

        if cfg.loss.ema_on:
            ema.update_params()     # apply ema

        # Show information
        if (batch_idx % cfg.train.show_log) == 0:
            msg = 'epoch: {}, batch: {}, train_eer: {:.4f}, train_loss: {:.4f}' \
                .format(epoch_idx, batch_idx, eer, loss.item())
            logger.info(msg)
        profiler.tick("Show information")

        # Plot snapshot
        if (batch_idx % cfg.train.plot_snapshot) == 0:
            plot_tool(cfg, log_file)
        profiler.tick("Plot snapshot")

        # Draw projections and save them to the backup folder
        # if umap_every != 0 and step % umap_every == 0:
        if (batch_idx % cfg.train.plot_umap) == 0:
            umap_dir = os.path.join(cfg.general.save_dir, 'umap')
            create_folder(umap_dir)

            projection_fpath = os.path.join(umap_dir, "umap_%06d.png" % (batch_idx))
            embeds = embeds.detach().cpu().numpy()
            draw_projections(cfg, embeds, batch_idx, projection_fpath)
        profiler.tick("Draw projections")

        # Save model
        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
            # if epoch_idx == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                if cfg.loss.ema_on:
                    ema.apply_shadow() # copy ema status to the model

                # save training model
                save_checkpoint(cfg, args.config_file, net, optimizer, epoch_idx, batch_idx)

                if cfg.general.is_test:
                    # TO
                    pass
                    # test(cfg, net, loss_func, epoch_idx, batch_idx,
                    #      logger, eval_validation_dataloader, mode='eval')
            
                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")

def main(): 
    parser = argparse.ArgumentParser(description='Streamax SV Training Engine')
    args = parser.parse_args()
    args.config_file = "/home/huanyuan/code/demo/Speech/SV/config/sv_config_TI_SV.py"
    train(args)


if __name__ == "__main__":
    main()