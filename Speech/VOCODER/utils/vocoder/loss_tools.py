import sys

sys.path.insert(0, '/home/huanyuan/code/demo/Speech')

from VOCODER.loss.vocoder import DiscriminatorAdversarialLoss
from VOCODER.loss.vocoder import FeatureMatchLoss
from VOCODER.loss.vocoder import GeneratorAdversarialLoss
from VOCODER.loss.vocoder import MelSpectrogramLoss
from VOCODER.loss.vocoder import MultiResolutionSTFTLoss


def load_criterion_wavegan(cfg):
    criterion = {}

    # GeneratorAdversarialLoss
    criterion["gen_adv"] = GeneratorAdversarialLoss( cfg.loss.generator_loss.average_by_discriminators,
                                                     cfg.loss.generator_loss.loss_type )
    criterion["gen_adv"].cuda()

    # DiscriminatorAdversarialLoss
    criterion["dis_adv"] = DiscriminatorAdversarialLoss( cfg.loss.discriminator_loss.average_by_discriminators,
                                                         cfg.loss.discriminator_loss.loss_type)
    criterion["dis_adv"].cuda()

    if cfg.loss.stft_loss.on:
        criterion["stft"] = MultiResolutionSTFTLoss( cfg.loss.stft_loss.fft_sizes, 
                                                     cfg.loss.stft_loss.hop_sizes,  
                                                     cfg.loss.stft_loss.win_lengths,  
                                                     cfg.loss.stft_loss.window )
        criterion["stft"].cuda()

    if cfg.loss.subband_stft_loss.on:
        raise NotImplementedError
    if cfg.loss.feat_match_loss.on:
        raise NotImplementedError
    if cfg.loss.mel_loss.on:
        raise NotImplementedError

    # TODO, NotImplementedError
    # define special module for subband processing
    if cfg.net.yaml["generator_params"]["out_channels"] > 1:
        # criterion["pqmf"] = PQMF(
        #     subbands=config["generator_params"]["out_channels"],
        #     # keep compatibility
        #     **config.get("pqmf_params", {}),
        # ).to(device)
        raise NotImplementedError

    return criterion


def calculate_loss_wavegan_generator(cfg, criterion, y, y_, p_, total_loss, mode='train'):
    # initialize
    gen_loss = 0.0

    # multi-resolution sfft loss
    if cfg.loss.stft_loss.on:
        sc_loss, mag_loss = criterion["stft"](y_, y)
        aux_loss = sc_loss + mag_loss
        total_loss["{}/spectral_convergence_loss".format(mode)] += sc_loss.item()
        total_loss["{}/log_stft_magnitude_loss".format(mode)] += mag_loss.item()

    # subband multi-resolution stft loss
    if cfg.loss.subband_stft_loss.on:
        raise NotImplementedError

    # mel spectrogram loss
    if cfg.loss.mel_loss.on:
        raise NotImplementedError
    
    # weighting aux loss
    gen_loss += cfg.loss.lambda_aux * aux_loss

    # adversarial loss
    if p_ is not None:
        adv_loss = criterion["gen_adv"](p_)
        total_loss["{}/adversarial_loss".format(mode)] += adv_loss.item()

        # feature matching loss
        if cfg.loss.feat_match_loss.on:
            raise NotImplementedError

        # add adversarial loss to generator loss
        gen_loss += cfg.loss.lambda_adv * adv_loss
        
    total_loss["{}/generator_loss".format(mode)] += gen_loss.item()
    return gen_loss


def calculate_loss_wavegan_discriminator(cfg, criterion, y, y_, p, p_, total_loss, mode='train'):
    dis_loss = 0.0

    # discriminator loss
    if y_ is not None and p is not None and p_ is not None:
        real_loss, fake_loss = criterion["dis_adv"](p_, p)
        dis_loss = real_loss + fake_loss
        total_loss["{}/real_loss".format(mode)] += real_loss.item()
        total_loss["{}/fake_loss".format(mode)] += fake_loss.item()
        total_loss["{}/discriminator_loss".format(mode)] += dis_loss.item()

    return dis_loss