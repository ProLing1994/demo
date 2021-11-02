import argparse
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/home/engineers/yh_rmai/code/demo/Speech')
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *
from Basic.utils.loss_tools import *
from Basic.utils.profiler_tools import *

from SV.utils.infer_tools import *

from TTS.dataset.sv2tts.text import *
from TTS.utils.sv2tts.visualizations_tools import *

from TTS.utils.vocoder.train_tools import *
import TTS.config.vocoder.hparams as hparams_vocoder
from TTS.dataset.vocoder.audio import *
from TTS.dataset.vocoder.distribution import *
from TTS.dataset.vocoder.vocoder_dataset_preload_audio_lmdb import prepare_data


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
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)

    # define loss function
    loss_func = loss_function(cfg) if net.module.mode == "RAW" else discretized_mix_logistic_loss

    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)
    scheduler = set_scheduler(cfg, optimizer)

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

    # 选择是否开启多说话人模式（SV2TTS）
    if cfg.general.mutil_speaker:
        msg = '训练模式：多说话人模式'
        logger.info(msg)
    else:
        msg = '训练模式：单说话人模式'
        logger.info(msg)

    # speaker verification net
    if cfg.general.mutil_speaker:
        cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
        sv_net = import_network(cfg_speaker_verification, 
                                cfg.speaker_verification.model_name, 
                                cfg.speaker_verification.class_name)
        if cfg.speaker_verification.load_mode_type == 0: 
            load_checkpoint(sv_net, cfg.speaker_verification.epoch, 
                            cfg.speaker_verification.model_dir)
        elif cfg.speaker_verification.load_mode_type == 1:
            load_checkpoint_from_path(sv_net, cfg.speaker_verification.model_path, 
                                        state_name='model_state',
                                        finetune_ignore_key_list=cfg.speaker_verification.ignore_key_list)
        else:
            raise Exception("[ERROR:] Unknow load mode type: {}".format(cfg.speaker_verification.load_mode_type))
        sv_net.eval()

    # synthesizer net
    cfg_synthesizer = load_cfg_file(cfg.synthesizer.config_file)
    synthesizer_net = import_network(cfg_synthesizer, cfg.synthesizer.model_name, cfg.synthesizer.class_name)
    if cfg.synthesizer.load_mode_type == 0: 
        load_checkpoint(synthesizer_net, cfg.synthesizer.epoch, 
                        cfg.synthesizer.model_dir)
    elif cfg.synthesizer.load_mode_type == 1:
        load_checkpoint_from_path(synthesizer_net, cfg.synthesizer.model_path, 
                                    state_name='model_state',
                                    finetune_ignore_key_list=cfg.synthesizer.ignore_key_list)
    else:
        raise Exception("[ERROR:] Unknow load mode type: {}".format(cfg.synthesizer.load_mode_type))
    synthesizer_net.eval()

    # define training dataset and testing dataset
    train_dataloader, len_train_dataset = generate_dataset(cfg, hparams.TRAINING_NAME)

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
        texts, text_lengths, mels, mel_lengths, embed_wavs, quant = data_iter.next()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # prepare embedding
        if cfg.general.mutil_speaker:
            embeds = []
            for embed_wav_idx in range(len(embed_wavs)):
                embed_wav = embed_wavs[embed_wav_idx]
                embed = embed_utterance(embed_wav, cfg_speaker_verification, sv_net)
                embeds.append(embed)
            embeds = torch.from_numpy(np.array(embeds))
        else: 
            embeds = None
        profiler.tick("prepare embedding")

        # prepare mel
        texts = texts.cuda()
        text_lengths = text_lengths.cuda()
        mels = mels.cuda()
        mel_lengths = mel_lengths.cuda()
        if cfg.general.mutil_speaker:
            embeds = embeds.cuda()
        else:
            embeds = None
        _, mel_out, _, _ = synthesizer_net(texts, text_lengths, mels, mel_lengths, embeds)
        profiler.tick("prepare mel")
        
        # prepare data
        mel_out = mel_out.detach().cpu().numpy()
        x, y, m, mel_list, quant_list = prepare_data(cfg, mel_out, mel_lengths, quant)
        x, m, y = x.cuda(), m.cuda(), y.cuda()
        profiler.tick("prepare data")

        # Forward pass
        y_hat = net(x, m)
        if isinstance(net, torch.nn.parallel.DataParallel):
            if net.module.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif net.module.mode == 'MOL':
                y = y.float()
        else:
            if net.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif net.mode == 'MOL':
                y = y.float()

        y = y.unsqueeze(-1)
        profiler.tick("Forward pass")

        # Calculate loss
        loss = loss_func(y_hat, y)
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
                    samples = 1
                    show_ressult(cfg, net, mel_list, quant_list, texts, samples, 
                                hparams_vocoder.voc_target, hparams_vocoder.voc_overlap, 
                                step=epoch_idx, save_dir=os.path.join(cfg.general.save_dir, 'wavs'))

                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")


def main(): 
    parser = argparse.ArgumentParser(description='Streamax SV2TTS Vocoder Training Engine')
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_vocoder_wavernn.py", nargs='?', help='config file')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/vocoder/tts_config_chinese_vocoder_wavernn.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()