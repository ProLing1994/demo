import argparse
from datetime import datetime
import os 
import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
# sys.path.insert(0, '/yuanhuan/code/demo/Speech')
from Basic.dataset import audio
from Basic.config import hparams
from Basic.utils.folder_tools import *
from Basic.utils.train_tools import *
from Basic.utils.loss_tools import *
from Basic.utils.profiler_tools import *

from SV.utils.infer_tools import *

from TTS.dataset.text.text import *
from TTS.utils.tts.train_tools import *
from TTS.utils.tts.visualizations_tools import *
from TTS.network.tts.guided_attention_loss import GuidedAttentionLoss
from TTS.utils.tts.infer_tools import *


sys.path.insert(0, '/home/huanyuan/code/demo/common')
from common.utils.python.logging_helpers import setup_logger


def show_ressult(cfg, attention, mel_prediction, target_spectrogram, input_seq, step,
                plot_dir, wav_dir, sample_num, loss):
    # create_folder
    create_folder(wav_dir)
    create_folder(plot_dir)

    # save text
    text = sequence_to_text(input_seq, lang = cfg.dataset.symbols_lang).split('~')[0]
    text_fpath = os.path.join(wav_dir, "text_step_{}_sample_{}.txt".format(step, sample_num))
    with open(text_fpath, "w") as f:
        f.write(text)

    # save some results for evaluation
    attention_path = os.path.join(plot_dir, "attention_step_{}_sample_{}.png".format(step, sample_num))
    save_attention(attention, attention_path)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = os.path.join(plot_dir, "mel_spectrogram_step_{}_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", datetime.now().strftime("%Y-%m-%d %H:%M"), step, loss)
    plot_spectrogram(mel_prediction, spec_fpath, title=title_str,
                    target_spectrogram=target_spectrogram,
                    max_len=target_spectrogram.shape[0])
    
    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = os.path.join(wav_dir, "mel_prediction_step_{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.compute_inv_mel_spectrogram(cfg, mel_prediction.T)
    wav_fpath = os.path.join(wav_dir, "wave_from_mel_step_{}_sample_{}.wav".format(step, sample_num))
    if len(wav):
        audio.save_wav(wav, str(wav_fpath), sampling_rate=cfg.dataset.sampling_rate)
    print("Input at step {}: {}".format(step, text))


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
    logger = setup_logger(log_file, 'tts_train')

    # define network
    net = import_network(cfg, cfg.net.model_name, cfg.net.class_name)
        
    # set training optimizer, learning rate scheduler
    optimizer = set_optimizer(cfg, net)
    scheduler = set_scheduler(cfg, optimizer)

    # define loss function
    net_loss = import_loss(cfg, cfg.net.model_name, cfg.net.loss_name)

    # ema
    if cfg.loss.ema_on:
        ema = EMA(net, 0.9999)

    # 选择是否开启引导者模式
    if cfg.guided_attn.on:
        msg = '引导者模式：学习 attention weight'
        logger.info(msg)
        attn_loss = GuidedAttentionLoss(sigma=cfg.guided_attn.attn_sigma, alpha=cfg.guided_attn.attn_lambda)

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

    if cfg.general.resume_epoch_num >= 0:
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

    # 选择是否开启多说话人模式（SV2TTS）
    if cfg.dataset.mutil_speaker:
        msg = '训练模式：多说话人模式'
        logger.info(msg)
    else:
        msg = '训练模式：单说话人模式'
        logger.info(msg)

    # 加载 sv model：引导者模式 和 多说话人模式
    if cfg.speaker_verification.on:
        # speaker verification net
        cfg_speaker_verification = load_cfg_file(cfg.speaker_verification.config_file)
        sv_net = import_network(cfg_speaker_verification, 
                                cfg.speaker_verification.model_name, 
                                cfg.speaker_verification.class_name)

        load_checkpoint(sv_net, 
                        cfg.speaker_verification.load_mode_type,
                        cfg.speaker_verification.model_dir, cfg.speaker_verification.epoch_num, cfg.speaker_verification.sub_folder_name,
                        cfg.speaker_verification.model_path,
                        cfg.speaker_verification.state_name, cfg.speaker_verification.ignore_key_list, cfg.speaker_verification.add_module_type)
        
        if cfg.speaker_verification.feedback_on:
            for param in sv_net.parameters():
                param.requires_grad = False
        else:
            sv_net.eval()

    # define training dataset and testing dataset
    # train_dataloader, len_train_dataset = generate_dataset_lmdb(cfg, hparams.TRAINING_NAME)
    train_dataloader, len_train_dataset = generate_dataset_hdf5(cfg, hparams.TRAINING_NAME)

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
        texts, text_lengths, mels, mel_lengths, stops, speaker_ids, embed_wavs = data_iter.next()

        # 选择是否开启多说话人模式（SV2TTS），需要计算 embedding
        if cfg.speaker_verification.on:
            embeds = []
            for embed_wav_idx in range(len(embed_wavs)):
                embed_wav = embed_wavs[embed_wav_idx]
                embed = embed_utterance(embed_wav, cfg_speaker_verification, sv_net)
                embeds.append(embed)
            embeds = torch.from_numpy(np.array(embeds))
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Data to device
        texts = texts.cuda()
        text_lengths = text_lengths.cuda()
        mels = mels.cuda()
        mel_lengths = mel_lengths.cuda()
        stops = stops.cuda()
        speaker_ids = speaker_ids.cuda()
        if cfg.speaker_verification.on:
            embeds = embeds.cuda()
        else:
            embeds = None
        profiler.tick("Data to device")
        
        # Forward pass
        # Parallelize model onto GPUS using workaround due to python bug
        if cfg.general.data_parallel_mode == 2 and cfg.general.num_gpus > 1:
            m1_hat, m2_hat, stop_pred, speaker_pred, attention = data_parallel_workaround(cfg, net, texts, text_lengths, mels, mel_lengths, speaker_ids, embeds)
        else:
            m1_hat, m2_hat, stop_pred, speaker_pred, attention = net(texts, text_lengths, mels, mel_lengths, speaker_ids, embeds)

        if cfg.speaker_verification.feedback_on:
            m2_embeds, mel_embeds = embed_mel(m2_hat, mels, mel_lengths, cfg, sv_net)
        profiler.tick("Forward pass")

        # Calculate loss
        predicits = (m1_hat, m2_hat, stop_pred, speaker_pred)
        targets = (mels, stops, speaker_ids)
        loss, m1_loss, m2_loss, stop_loss, speaker_loss = net_loss(predicits, targets)
        
        # Calculate guided_attn_loss
        if cfg.guided_attn.on:
            if isinstance(net, torch.nn.parallel.DataParallel):
                reduction_factor = net.module.reduction_factor()
            else:
                reduction_factor = net.reduction_factor()

            # NOTE: length of output for auto-regressive
            # input will be changed when r > 1
            if reduction_factor > 1:
                mel_lengths_in = mel_lengths.new([olen // reduction_factor for olen in mel_lengths])
            else:
                mel_lengths_in = mel_lengths
            guided_attn_loss = attn_loss(attention, text_lengths, mel_lengths_in)
            loss += 1.0 * guided_attn_loss

        # Calculate feedback_loss
        if cfg.speaker_verification.feedback_on:
            if cfg.speaker_verification.embed_loss_func == 'cos':
                feedback_loss = 1 - torch.cosine_similarity(m2_embeds, mel_embeds, dim=1)
                feedback_loss = feedback_loss.sum()
                feedback_loss = cfg.speaker_verification.embed_loss_scale * feedback_loss
            elif cfg.speaker_verification.embed_loss_func == 'mse':
                feedback_loss = F.mse_loss(m2_embeds, mel_embeds)
                feedback_loss = cfg.speaker_verification.embed_loss_scale * feedback_loss
            loss += 1.0 * feedback_loss
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
            msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, m1_loss: {:.4f}, m2_loss: {:.4f}, stop_loss: {:.4f}'.format(
                    epoch_idx, batch_idx, loss.item(), m1_loss.item(), m2_loss.item(), stop_loss.item())
            if cfg.guided_attn.on:
                msg += ', guided_attn_loss: {:.4f}'.format(guided_attn_loss.item())
            if not speaker_loss is None:
                msg += ', speaker_loss: {:.4f}'.format(speaker_loss.item())
            if cfg.speaker_verification.feedback_on:
                msg += ', feedback_loss: {:.4f}'.format(feedback_loss.item())
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
                    # show result
                    sample_idx = 0
                    mel_prediction = m2_hat[sample_idx].detach().cpu().numpy().T
                    target_spectrogram = mels[sample_idx].detach().cpu().numpy().T
                    mel_length = mel_prediction.shape[0]
                    attention_len = mel_length // cfg.net.r 
                        
                    attention_prediction = attention[sample_idx][:, :attention_len].detach().cpu().numpy()
                    target_text = texts[sample_idx].detach().cpu().numpy()
                    show_ressult(cfg, 
                                attention=attention_prediction,
                                mel_prediction=mel_prediction,
                                target_spectrogram=target_spectrogram,
                                input_seq=target_text,
                                step=epoch_idx,
                                plot_dir=os.path.join(cfg.general.save_dir, 'plots'),
                                wav_dir=os.path.join(cfg.general.save_dir, 'wavs'),
                                sample_num=sample_idx + 1,
                                loss=loss)

                if cfg.loss.ema_on:
                    ema.restore() # resume the model parameters
        profiler.tick("Save model")


def main(): 
    parser = argparse.ArgumentParser(description='Streamax TTS Training Engine')
    # parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_english_sv2tts.py", nargs='?', help='config file')
    parser.add_argument('-i', '--config_file', type=str, default="/home/huanyuan/code/demo/Speech/TTS/config/tts/tts_config_chinese_sv2tts.py", nargs='?', help='config file')
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()