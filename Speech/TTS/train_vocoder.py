import argparse
from datetime import datetime
import os 
import sys
from tqdm import tqdm

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *
from Basic.utils.loss_tools import *
from Basic.utils.profiler_tools import *

from SV.utils.infer_tools import *
from TTS.utils.vocoder.train_tools import *

sys.path.insert(0, '/home/huanyuan/code/demo/common')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/common')
from common.utils.python.logging_helpers import setup_logger


def prepare_data(cfg, mel_out, mel_frames, quant):
    for idx in range(len(quant)):
        quant_idx = quant[idx]
        mel_out_idx = mel_out[idx]
    print()
    pass

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
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # define loss function
    loss_func = loss_function(cfg)

    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)
    scheduler = set_scheduler(cfg, optimizer)

    # ema
    if cfg.loss.ema_on:
        ema = EMA(net, 0.9999)

    # load checkpoint if finetune_on == True or resume epoch > 0
    if cfg.general.finetune_on == True:
        # fintune, Load model, reset learning rate
        if cfg.general.finetune_model_path == "":
            load_checkpoint(net, cfg.general.finetune_epoch, 
                            cfg.general.finetune_model_dir, 
                            sub_folder_name='pretrain_model')
        # fintune,
        else:
            load_checkpoint_from_path(net, cfg.general.finetune_model_path, 
                                        state_name='model_state',
                                        finetune_ignore_key_list=cfg.general.finetune_ignore_key_list)
        start_epoch, start_batch = 0, 0
        last_save_epoch = 0

    if cfg.general.resume_epoch >= 0:
        # resume, Load the model, continue the previous learning rate
        start_epoch, start_batch = load_checkpoint(net, cfg.general.resume_epoch,
                                                    cfg.general.save_dir, 
                                                    optimizer=optimizer)
        last_save_epoch = start_epoch
    else:
        start_epoch, start_batch = 0, 0
        last_save_epoch = 0

    # speaker verification net
    cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
    sv_net = import_network(cfg_speaker_verification, 
                            cfg.speaker_verification.model_name, 
                            cfg.speaker_verification.class_name,
                            cfg.speaker_verification.model_prefix_name)
    if cfg.speaker_verification.model_path == "":
        load_checkpoint(sv_net, cfg.speaker_verification.epoch, 
                        cfg.speaker_verification.model_dir)
    else:
        load_checkpoint_from_path(sv_net, cfg.speaker_verification.model_path, 
                                    state_name='model_state',
                                    finetune_ignore_key_list=cfg.speaker_verification.ignore_key_list)
    sv_net.eval()

    # synthesizer net
    cfg_synthesizer = load_cfg_file(cfg.synthesizer.config_file)
    synthesizer_net = import_network(cfg_synthesizer, cfg.synthesizer.model_name, cfg.synthesizer.class_name)
    synthesizer_net.r = cfg_synthesizer.net.r 
    if cfg.synthesizer.model_path == "":
        load_checkpoint(synthesizer_net, cfg.synthesizer.epoch, 
                        cfg.synthesizer.model_dir)
    else:
        load_checkpoint_from_path(synthesizer_net, cfg.synthesizer.model_path, 
                                    state_name='model_state',
                                    finetune_ignore_key_list=cfg.synthesizer.ignore_key_list)
    synthesizer_net.eval()

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
        texts, mels, mel_frames, embed_wavs, quant = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # prepare embedding
        embeds = []
        for embed_wav_idx in range(len(embed_wavs)):
            embed_wav = embed_wavs[embed_wav_idx]
            embed = embed_utterance(embed_wav, cfg_speaker_verification, sv_net)
            embeds.append(embed)

        embeds = torch.from_numpy(np.array(embeds))
        profiler.tick("prepare embedding")

        # prepare mel
        texts = texts.cuda()
        mels = mels.cuda()
        embeds = embeds.cuda()
        _, mel_out, _, _ = synthesizer_net(texts, mels, embeds)
        profiler.tick("prepare mel")
        
        # prepare data
        mel_out = mel_out.detach().cpu().numpy()
        prepare_data(cfg, mel_out, mel_frames, quant)
        profiler.tick("prepare data")

        # Forward pass
        
        profiler.tick("Forward pass")

        # Calculate loss
        m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
        m2_loss = F.mse_loss(m2_hat, mels)
        stop_loss = F.binary_cross_entropy(stop_pred, stop)
        loss = m1_loss + m2_loss + stop_loss
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
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}'.format(epoch_idx, batch_idx, loss.item())
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
                save_checkpoint(cfg, args.config_file, net, optimizer, epoch_idx, batch_idx)

                if cfg.general.is_test:
                    # To Do
                    pass

                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")


def main(): 
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Vocoder Training Engine')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_vocoder_wavernn.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()