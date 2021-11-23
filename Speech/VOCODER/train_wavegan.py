import argparse
from collections import defaultdict
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
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
import VOCODER.config.vocoder.hparams as hparams_vocoder
from VOCODER.dataset.vocoder.audio import *
from VOCODER.dataset.vocoder.distribution import *
from VOCODER.dataset.vocoder.vocoder_dataset_preload_audio_lmdb import prepare_data


sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
from common.utils.python.logging_helpers import setup_logger


def show_ressult(cfg, net, mel_list, quant_list, texts, samples, 
                    target, overlap, 
                    step, save_dir):
    # create_folder
    create_folder(save_dir)

    for idx in range(samples):
        # init 
        bits = 16 if hparams_vocoder.voc_mode == 'MOL' else hparams_vocoder.voc_bits

        # text
        texts = texts[idx].detach().cpu().numpy()
        target_text = sequence_to_text(texts).split('~')[0]
        text_fpath = os.path.join(save_dir, "text_step_{}_sample_{}.txt".format(step, idx))
        with open(text_fpath, "w") as f:
            f.write(target_text)

        # wav_target
        wav_target = quant_list[idx]
        if hparams_vocoder.mu_law and hparams_vocoder.voc_mode != 'MOL' :
            wav_target = decode_mu_law(wav_target, 2**bits, from_labels=True)
        else :
            wav_target = label_2_float(wav_target, bits) 
        wav_fpath = os.path.join(save_dir, "wav_target_step_{}_sample_{}.wav".format(step, idx))
        audio.save_wav(wav_target, wav_fpath, sr=cfg.dataset.sample_rate)

        # wav_forward
        mel_forward = torch.tensor(mel_list[idx]).unsqueeze(0)

        for bool_gen_batched in [True, False]:
            print('\n| Generating: {}/{}, bool_gen_batched: {}'.format(idx, samples, bool_gen_batched))

            if isinstance(net, torch.nn.parallel.DataParallel):
                wav_forward = net.module.generate(mel_forward, bool_gen_batched, target, overlap, hparams_vocoder.mu_law)
            else:
                wav_forward = net.generate(mel_forward, bool_gen_batched, target, overlap, hparams_vocoder.mu_law)

            batch_str = "gen_batched_target_%d_overlap_%d" % (target, overlap) if bool_gen_batched else \
                "gen_not_batched"
            wav_forward_fpath = os.path.join(save_dir, "wav_forward_step_{}_sample_{}_{}.wav".format(step, idx, batch_str))
            audio.save_wav(wav_forward, wav_forward_fpath, sr=cfg.dataset.sample_rate)


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

    # loop over batches
    for i in range(batch_number):
        for key in model.keys():
            model[key].train()

        total_train_loss = defaultdict(float)
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
                msg += ', {} = {:.4f}'.format(str(key), total_train_loss[key])
            logger.info(msg)
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

        # # Test model
        # if epoch_idx % cfg.train.save_epochs == 0 or epoch_idx == cfg.train.num_epochs - 1:
        #     if last_test_epochh != epoch_idx and cfg.general.is_test:
        #         last_test_epochh = epoch_idx

        #         samples = 1
        #         show_ressult(cfg, net, mel_list, quant_list, texts, samples, 
        #                     hparams_vocoder.voc_target, hparams_vocoder.voc_overlap, 
        #                     step=epoch_idx, save_dir=os.path.join(cfg.general.save_dir, 'wavs'))



def main(): 
    parser = argparse.ArgumentParser(description='Streamax TTS Vocoder Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/VOCODER/config/vocoder/vocoder_config_chinese_wavegan.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()