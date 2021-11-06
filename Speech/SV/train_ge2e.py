import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.loss_tools import *
from Basic.utils.profiler_tools import *
from Basic.utils.train_tools import *

from SV.utils.loss_tools import *
from SV.utils.train_tools import *
from SV.utils.visualizations_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
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

    embeds = []
    scores = []
    labels = []
    losses = []
    for _, (input, label) in tqdm(enumerate(test_data_loader)):
        input = input.cuda()
        label = label.cuda()

        if cfg.dataset.h_alignment == True:
            hisi_input = input[:, :, :(input.shape[2] // 16) * 16, :]
            if cfg.loss.method == 'ge2e':
                embed = net(hisi_input)
            elif cfg.loss.method == 'softmax':
                embed, score = net(hisi_input)
        else:
            if cfg.loss.method == 'ge2e':
                embed = net(input)
            elif cfg.loss.method == 'softmax':
                embed, score = net(input)

        embeds.append(embed.detach().cpu().numpy())
        if cfg.loss.method == 'softmax':
            loss = loss_func(score, label)
            scores.append(torch.max(score, 1)[1].cpu().data.numpy())
            labels.append(label.cpu().data.numpy())
            losses.append(loss.item())

    # Calculate loss
    embeds_np = np.array(embeds)
    if isinstance(net, torch.nn.parallel.DataParallel):
        sim_matrix = net.module.similarity_matrix_cpu(embeds_np)
    else:
        sim_matrix = net.similarity_matrix_cpu(embeds_np)

    if cfg.loss.method == 'ge2e':
        eer = compute_eer(embeds_np, sim_matrix)
        loss = -1.0
    elif cfg.loss.method == 'softmax':
        eer = compute_eer(embeds_np, sim_matrix)
        loss = np.array(losses).sum()/float(len(labels))
    
    # Caltulate accuracy
    if cfg.loss.method == 'ge2e':
        accuracy = -1.0
    elif cfg.loss.method == 'softmax':
        accuracy = float((np.array(scores) == np.array(labels)
                        ).astype(int).sum()) / float(len(labels))

    # Show information
    msg = 'epoch: {}, batch: {}, {}_accuracy: {:.4f}, {}_eer: {:.4f}, {}_loss: {:.4f}'.format(
        epoch_idx, batch_idx, mode, accuracy, mode, eer, mode, loss)
    logger.info(msg)

    # Draw projections and save them to the backup folder
    umap_dir = os.path.join(cfg.general.save_dir, 'umap')
    create_folder(umap_dir)
    projection_fpath = os.path.join(umap_dir, "umap_testing_%05d.png" % (epoch_idx))
    draw_projections(embeds_np, epoch_idx, projection_fpath)


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
        if cfg.general.load_mode_type == 0: 
            load_checkpoint(net, cfg.general.finetune_epoch, 
                            cfg.general.finetune_model_dir, 
                            sub_folder_name='pretrain_model')
        # fintune, 
        elif cfg.general.load_mode_type == 1:
            load_checkpoint_from_path(net, cfg.general.finetune_model_path, 
                                        state_name=cfg.general.finetune_model_state,
                                        finetune_ignore_key_list=cfg.general.finetune_ignore_key_list)
        else:
            raise Exception("[ERROR:] Unknow load mode type: {}".format(cfg.general.load_mode_type))
        start_epoch, start_batch = 0, 0
        last_save_epoch, last_plot_epoch = 0, 0
    if cfg.general.resume_epoch >= 0:
        # resume, Load the model, continue the previous learning rate
        start_epoch, start_batch = load_checkpoint(net, cfg.general.resume_epoch,
                                                        cfg.general.save_dir, 
                                                        optimizer=optimizer)
        last_save_epoch = start_epoch
        last_plot_epoch = start_epoch
    else:
        start_epoch, start_batch = 0, 0
        last_save_epoch, last_plot_epoch = 0, 0

    # knowledge distillation
    if cfg.knowledge_distillation.on:
        msg = 'Knowledge Distillation: {} -> {}'.format(cfg.knowledge_distillation.teacher_model_name, cfg.net.model_name)
        logger.info(msg)

        teacher_model = import_network(cfg, cfg.knowledge_distillation.teacher_model_name, 
                                        cfg.knowledge_distillation.teacher_class_name)
        load_checkpoint(teacher_model, cfg.knowledge_distillation.epoch, 
                        cfg.knowledge_distillation.teacher_model_dir)
        teacher_model.eval()

    # define training dataset and testing dataset
    train_dataloader, len_train_dataset = generate_dataset(cfg, hparams.TRAINING_NAME)
    if cfg.general.is_test:
        testing_dataloader = generate_test_dataset(cfg, hparams.TESTING_NAME)

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
        inputs, labels = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Data to device
        inputs = inputs.cuda()
        labels = labels.cuda()
        profiler.tick("Data to device")
        
        # Forward pass
        if cfg.dataset.h_alignment == True:
            hisi_input = inputs[:, :, :(inputs.shape[2] // 16) * 16, :]
            if cfg.loss.method == 'ge2e':
                embeds = net(hisi_input)
            elif cfg.loss.method == 'softmax':
                embeds, scores = net(hisi_input)
        else:
            if cfg.loss.method == 'ge2e':
                embeds = net(inputs)
            elif cfg.loss.method == 'softmax':
                embeds, scores = net(inputs)
        profiler.tick("Forward pass")

        if isinstance(net, torch.nn.parallel.DataParallel):
            embeds = net.module.embeds_view(embeds)
            sim_matrix = net.module.similarity_matrix(embeds)
        else:
            embeds = net.embeds_view(embeds)
            sim_matrix = net.similarity_matrix(embeds)
        
        # Calculate loss
        if cfg.loss.method == 'ge2e':
            loss, eer = ge2e_loss(embeds, sim_matrix, loss_func)
        elif cfg.loss.method == 'softmax':
            _, eer = ge2e_loss(embeds, sim_matrix, loss_func)
            loss = loss_func(scores, labels)
        if cfg.knowledge_distillation.on:
            teacher_model.eval()
            teacher_embeds, _ = teacher_model(inputs)
            # TO DO
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

        # Caltulate accuracy
        if cfg.loss.method == 'ge2e':
            accuracy = -1.0
        elif cfg.loss.method == 'softmax':
            pred_y = torch.max(scores, 1)[1].cpu().data.numpy()
            accuracy = float((pred_y == labels.cpu().data.numpy()).astype(
                int).sum()) / float(labels.size(0))
        profiler.tick("Caltulate accuracy")

        # Show information
        if (batch_idx % cfg.train.show_log) == 0:
            msg = 'epoch: {}, batch: {}, train_accuracy: {:.4f}, train_eer: {:.4f}, train_loss: {:.4f}' \
                .format(epoch_idx, batch_idx, accuracy, eer, loss.item())
            logger.info(msg)
        profiler.tick("Show information")

        # Plot snapshot
        if (batch_idx % cfg.train.plot_snapshot) == 0:
            plot_tool(cfg, log_file)
        profiler.tick("Plot snapshot")

        # Draw projections and save them to the backup folder
        if (epoch_idx % cfg.train.plot_umap) == 0:
            if last_plot_epoch != epoch_idx:
                last_plot_epoch = epoch_idx

                umap_dir = os.path.join(cfg.general.save_dir, 'umap')
                create_folder(umap_dir)
                projection_fpath = os.path.join(umap_dir, "umap_training_%05d.png" % (epoch_idx))
                embeds_cpu = embeds.detach().cpu().numpy()
                draw_projections(embeds_cpu, epoch_idx, projection_fpath)
        profiler.tick("Draw projections")

        # Save model
        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                if cfg.loss.ema_on:
                    ema.apply_shadow() # copy ema status to the model

                # save training model
                save_checkpoint(cfg, args.config_file, net, optimizer, epoch_idx, batch_idx)

                if cfg.general.is_test:
                    test(cfg, net, loss_func, epoch_idx, batch_idx,
                         logger, testing_dataloader, mode='eval')
            
                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")

def main(): 
    parser = argparse.ArgumentParser(description='Streamax SV Training Engine')
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_english_TI_SV.py", nargs='?', help='config file')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/SV/config/sv_config_chinese_TI_SV.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()