import sys
import torch

# sys.path.insert(0, '/home/huanyuan/code/demo/Speech')
sys.path.insert(0, '/yuanhuan/code/demo/Speech')
import VOCODER.optimizers.vocoder

def load_optimizers_wavegan(cfg, model):
    optimizer = {}

    if cfg.optimizer.generator_optimizer.type == "RAdam":
        optimizer['generator'] = VOCODER.optimizers.vocoder.RAdam( model["generator"].parameters(), 
                                                                    cfg.optimizer.generator_optimizer.lr ,
                                                                    cfg.optimizer.generator_optimizer.betas ,
                                                                    cfg.optimizer.generator_optimizer.eps ,
                                                                    cfg.optimizer.generator_optimizer.weight_decay )
    else:
        raise NotImplementedError

    if cfg.optimizer.discriminator_optimizer.type == "RAdam":
        optimizer['discriminator'] = VOCODER.optimizers.vocoder.RAdam( model["discriminator"].parameters(),  
                                                                        cfg.optimizer.discriminator_optimizer.lr ,
                                                                        cfg.optimizer.discriminator_optimizer.betas ,
                                                                        cfg.optimizer.discriminator_optimizer.eps ,
                                                                        cfg.optimizer.discriminator_optimizer.weight_decay )
    else:
        raise NotImplementedError

    return optimizer


def load_scheduler_wavegan(cfg, optimizer):
    scheduler = {}
    
    if cfg.optimizer.generator_scheduler.type  == 'StepLR':
        scheduler["generator"] = torch.optim.lr_scheduler.StepLR(optimizer=optimizer['generator'], 
                                                    step_size=cfg.optimizer.generator_scheduler.step_size, 
                                                    gamma=cfg.optimizer.generator_scheduler.gamma)
    else:
        raise NotImplementedError

    if cfg.optimizer.discriminator_scheduler.type  == 'StepLR':
        scheduler["discriminator"] = torch.optim.lr_scheduler.StepLR(optimizer=optimizer['discriminator'], 
                                                    step_size=cfg.optimizer.discriminator_scheduler.step_size, 
                                                    gamma=cfg.optimizer.discriminator_scheduler.gamma)
    else:
        raise NotImplementedError

    return scheduler