import torch


class Optimizer(object):
    def __init__(self, config, net):
        self.config = config
        self.optimizer = set_optimizer(self.config, net)
        self.scheduler = set_scheduler(self.config, self.optimizer)
        self.global_step = 1

    def step(self):
        self.global_step += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def decay_lr(self):
        self.lr *= self.decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def update_scheduler(self, epoch_idx):
        update_scheduler(self.config, self.scheduler, epoch_idx)

def set_optimizer(cfg, net):
    """
    :param cfg:   training configure file
    :param net:   pytorch network
    :return:
    """
    if cfg.train.optimizer == 'SGD':
        opt = torch.optim.SGD(net.parameters(),
                              lr=cfg.train.lr,
                              momentum=cfg.train.momentum,
                              weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'Adam':
        opt = torch.optim.Adam(net.parameters(),
                               lr=cfg.train.lr,
                               betas=cfg.train.betas,
                               weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError('Unknown loss optimizer')

    return opt

def set_scheduler(cfg, optimizer):
    """
    :param cfg:   training configure file
    :param optimizer:   pytorch optimizer
    :return:
    """
    scheduler = None
    if cfg.train.optimizer == 'SGD':
        if cfg.train.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                        step_size=cfg.train.lr_step_size, 
                                                        gamma=cfg.train.lr_gamma)
        elif cfg.train.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, 
                                                        T_0=cfg.train.T_0,
                                                        T_mult=cfg.train.T_mult)
        else:
            raise ValueError('Unknown loss scheduler')
    return scheduler

def update_scheduler(cfg, scheduler, epoch_idx):
    """
    :param cfg:   training configure file
    :param scheduler:   pytorch scheduler
    :param epoch_idx:   
    :return:
    """
    if cfg.train.optimizer == 'SGD':
        scheduler.step(epoch_idx)
    else:
        pass