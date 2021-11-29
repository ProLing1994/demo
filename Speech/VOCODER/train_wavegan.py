import argparse
from collections import defaultdict
import os 
import sys
import soundfile as sf

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *
from Basic.utils.profiler_tools import *

from SV.utils.infer_tools import *

from TTS.dataset.text.text import *
from TTS.utils.tts.visualizations_tools import *

from VOCODER.utils.vocoder.train_tools import *
from VOCODER.utils.vocoder.loss_tools import *
from VOCODER.utils.vocoder.optimizers_tools import *
from VOCODER.dataset.vocoder.audio import *
from VOCODER.dataset.vocoder.distribution import *
from VOCODER.dataset.vocoder.vocoder_dataset_preload_audio_lmdb import prepare_data


sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/yuanhuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
from common.utils.python.logging_helpers import setup_logger


def test(cfg, model, criterion, x, y, logger, epoch_idx, batch_idx):
    # total_eval_loss
    total_eval_loss = defaultdict(float)

    # change mode
    for key in model.keys():
        model[key].eval()

    #######################
    #      Generator      #
    #######################
    y_ = model["generator"](*x)
    if cfg.net.yaml["generator_params"]["out_channels"] > 1:
        # y_mb_ = y_
        # y_ = criterion["pqmf"].synthesis(y_mb_)
        raise NotImplementedError

    #######################
    #    Discriminator    #
    #######################
    p = model["discriminator"](y)
    p_ = model["discriminator"](y_)

    calculate_loss_wavegan_generator(cfg, criterion, y, y_, p_, total_eval_loss, mode='eval')
    calculate_loss_wavegan_discriminator(cfg, criterion, y, y_, p, p_, total_eval_loss, mode='eval')

    # Show information
    msg = 'epoch: {}, batch: {}'.format(epoch_idx, batch_idx)
    for key in total_eval_loss.keys():
        msg += ', {}:{:.4f}'.format(str(key), total_eval_loss[key])
    logger.info(msg)

    # create_folder
    save_dir = os.path.join(cfg.general.save_dir, 'wavs')
    create_folder(save_dir)

    for idx, (y_idx, y_idx_) in enumerate(zip(y, y_)):
        y_idx, y_idx_ = y_idx.view(-1).cpu().detach().numpy(), y_idx_.view(-1).cpu().detach().numpy()
        
        # 预加重
        if cfg.dataset.compute_mel_type == "fbank_preemphasis_log_manual":
            y_idx = audio.compute_de_emphasis(y_idx)
            y_idx_ = audio.compute_de_emphasis(y_idx_)

        # plot_figure
        figname = os.path.join(save_dir, "plot_step_{}_{}.png".format(epoch_idx, idx))
        plt.subplot(2, 1, 1)
        plt.plot(y_idx)
        plt.title("groundtruth speech")
        plt.subplot(2, 1, 2)
        plt.plot(y_idx_)
        plt.title(f"generated speech @ {epoch_idx} steps")
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()

        # wav_target
        y_idx = np.clip(y_idx, -1, 1)
        wav_fpath = os.path.join(save_dir, "wav_target_step_{}_{}.wav".format(epoch_idx, idx))
        sf.write(wav_fpath, y_idx, cfg.dataset.sampling_rate, "PCM_16")

        # wav_forward
        y_idx_ = np.clip(y_idx_, -1, 1)
        wav_forward_fpath = os.path.join(save_dir, "wav_forward_step_{}_{}.wav".format(epoch_idx, idx))
        sf.write(wav_forward_fpath, y_idx_, cfg.dataset.sampling_rate, "PCM_16",)

    # restore mode
    for key in model.keys():
        model[key].train()


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
    logger = setup_logger(log_file, 'vocoder_train')

    # define network
    model = {}
    model['generator'] = import_network(cfg, cfg.net.generator_model_name, cfg.net.generator_class_name)
    model['discriminator'] = import_network(cfg, cfg.net.discriminator_model_name, cfg.net.discriminator_class_name)

    # define loss function
    criterion = load_criterion_wavegan(cfg) 
    
    # set training optimizer, learning rate scheduler
    optimizer = load_optimizers_wavegan(cfg, model)
    scheduler = load_scheduler_wavegan(cfg, optimizer)

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        load_checkpoint_wavegan(cfg.general.finetune_model_path, model)

    if cfg.general.resume_on == True:
        # resume, Load the model, continue the previous learning rate
        start_epoch, start_batch = load_checkpoint_wavegan(cfg.general.resume_model_path, model, optimizer=optimizer, scheduler=scheduler)
        last_save_epoch, last_test_epoch = start_epoch, start_epoch
    else:
        start_epoch, start_batch = 0, 0
        last_save_epoch, last_test_epoch = 0, 0

    # define training dataset and testing dataset
    train_dataloader, len_train_dataset = generate_dataset_wavegan(cfg, hparams.TRAINING_NAME)

    msg = 'Training dataset number: {}'.format(len_train_dataset)
    logger.info(msg)

    batch_number = len(train_dataloader)
    data_iter = iter(train_dataloader)
    batch_idx = start_batch

    # profiler
    profiler = Profiler(summarize_every=cfg.train.show_log, disabled=False)

    # total_train_loss
    total_train_loss = defaultdict(float)

    # loop over batches
    for i in range(batch_number):
        for key in model.keys():
            model[key].train()

        epoch_idx = start_epoch + i * cfg.train.batch_size // len_train_dataset
        batch_idx += 1

        # Blocking, waiting for batch (threaded)
        x, y = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # prepare data
        x = tuple([x_.cuda() for x_ in x])
        y = y.cuda()
        profiler.tick("prepare data")

        #######################
        #      Generator      #
        #######################
        # Forward pass
        if batch_idx >= cfg.train.generator_train_start_steps:
            y_ = model["generator"](*x)

            # reconstruct the signal from multi-band signal
            if cfg.net.yaml["generator_params"]["out_channels"] > 1:
                # y_mb_ = y_
                # y_ = criterion["pqmf"].synthesis(y_mb_)
                raise NotImplementedError

        if batch_idx >= cfg.train.discriminator_train_start_steps:
            p_ = model["discriminator"](y_)
        else:
            p_ = None
        profiler.tick("Generator Forward pass")

        # Calculate loss
        if batch_idx >= cfg.train.generator_train_start_steps:
            gen_loss = calculate_loss_wavegan_generator(cfg, criterion, y, y_, p_, total_train_loss)
        profiler.tick("Generator Calculate Loss")
                    
        # Backward pass
        if batch_idx >= cfg.train.generator_train_start_steps:
            # model["generator"].zero_grad()
            optimizer["generator"].zero_grad()
            gen_loss.backward()
        profiler.tick("Generator Backward pass")

        # Parameter update
        if batch_idx >= cfg.train.generator_train_start_steps:
            if cfg.optimizer.generator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model["generator"].parameters(), cfg.optimizer.generator_grad_norm)
            optimizer["generator"].step()
            scheduler["generator"].step()
        profiler.tick("Generator Parameter update")

        #######################
        #    Discriminator    #
        #######################
        # Forward pass
        if batch_idx >= cfg.train.discriminator_train_start_steps:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = model["generator"](*x)
            if cfg.net.yaml["generator_params"]["out_channels"] > 1:
                # y_ = criterion["pqmf"].synthesis(y_)
                raise NotImplementedError

            p = model["discriminator"](y)
            p_ = model["discriminator"](y_.detach())
        profiler.tick("Discriminator Forward pass")

        # Calculate loss
        if batch_idx >= cfg.train.discriminator_train_start_steps:
            dis_loss = calculate_loss_wavegan_discriminator(cfg, criterion, y, y_, p, p_, total_train_loss)
        profiler.tick("Discriminator Calculate Loss")

        # Backward pass
        if batch_idx >= cfg.train.discriminator_train_start_steps:
            # model["discriminator"].zero_grad()
            optimizer["discriminator"].zero_grad()
            dis_loss.backward()
        profiler.tick("Discriminator Backward pass")

        # Parameter update
        if batch_idx >= cfg.train.discriminator_train_start_steps:
            if cfg.optimizer.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model["discriminator"].parameters(), cfg.optimizer.discriminator_grad_norm)
            optimizer["discriminator"].step()
            scheduler["discriminator"].step()
        profiler.tick("Discriminator Parameter update")

        # Show information
        if (batch_idx % cfg.train.show_log) == 0:
            msg = 'epoch: {}, batch: {}'.format(epoch_idx, batch_idx)
            for key in total_train_loss.keys():
                total_train_loss[key] /= cfg.train.show_log
                msg += ', {}:{:.4f}'.format(str(key), total_train_loss[key])
            logger.info(msg)

            # reset
            total_train_loss = defaultdict(float)
        profiler.tick("Show information")

        # Plot snapshot
        if (batch_idx % cfg.train.plot_snapshot) == 0:
            plot_tool_wavegan(cfg, log_file)
        profiler.tick("Plot snapshot")

        # Save model
        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_save_epoch != epoch_idx:
                last_save_epoch = epoch_idx

                # save training model
                save_checkpoint_wavegan(cfg, args.config_file, model, optimizer, scheduler, epoch_idx, batch_idx)
        profiler.tick("Save model")

        # Test model
        if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
            if last_test_epoch != epoch_idx and cfg.general.is_test:
                last_test_epoch = epoch_idx

                test(cfg, model, criterion, x, y, logger, epoch_idx, batch_idx)



def main(): 
    parser = argparse.ArgumentParser(description='Streamax TTS Vocoder Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VOCODER/config/vocoder/vocoder_config_chinese_wavegan.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()