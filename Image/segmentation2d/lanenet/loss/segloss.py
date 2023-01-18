"""Modified from http://192.168.80.93/jhwen/segmentation/tree/branch_v2.0
"""

from loss.dice_loss import (DC_and_CE_loss, DC_and_topk_loss, DC_and_Focal_loss,
                                GDL_and_CE_loss, GDL_and_topk_loss, 
                                DC_and_BD_loss, DC_and_HD_loss, DC_FC_CE_loss)

class DC_CE_loss(DC_and_CE_loss):
    """DC_CE_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 smooth=1.,
                 class_weight=None, 
                 ignore_index=255, 
                 square_dice=False,
                 weight_ce=1, 
                 weight_dice=1,
                 log_dice=False, 
                 loss_weight=1.0,
                 **kwards):

        self.loss_weight = loss_weight
        super(DC_CE_loss, self).__init__({'batch_dice': batch_dice, 'smooth': smooth, 'do_bg': do_bg}, 
                                         {'weight': class_weight},
                                         weight_ce=weight_ce, 
                                         weight_dice=weight_dice,
                                         log_dice=log_dice,
                                         ignore_label=ignore_index, 
                                         square_dice=square_dice)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(DC_CE_loss, self).forward(seg_logit, seg_label)

class DC_Topk_loss(DC_and_topk_loss):
    """DC_Topk_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 smooth=1.,
                 class_weight=None, 
                 ignore_index=255, 
                 k=10, 
                 aggregate="sum", 
                 square_dice=False,
                 loss_weight=1.0,
                 **kwards):

        self.loss_weight = loss_weight
        super(DC_Topk_loss, self).__init__({'batch_dice': batch_dice, 'smooth': smooth, 'do_bg': do_bg}, 
                                           {'weight': class_weight, 'ignore_index': ignore_index, 'k': k},
                                           aggregate=aggregate, 
                                           square_dice=square_dice)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(DC_Topk_loss, self).forward(seg_logit, seg_label)

class DC_Focal_loss(DC_and_Focal_loss):
    """DC_Focal_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 dice_smooth=1.,
                 alpha=None, 
                 gamma=2, 
                 balance_index=0, 
                 focal_smooth=1e-5, 
                 loss_weight=1.0,
                 **kwards):

        self.loss_weight = loss_weight
        super(DC_Focal_loss, self).__init__({'batch_dice': batch_dice, 'smooth': dice_smooth, 'do_bg': do_bg}, 
                                            {'alpha': alpha, 'gamma': gamma, 'balance_index': balance_index, 'smooth': focal_smooth},)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(DC_Focal_loss, self).forward(seg_logit, seg_label)

class GDL_CE_loss(GDL_and_CE_loss):
    """GDL_CE_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 dice_smooth=1.,
                 square_volumes=False,
                 class_weight=None,
                 ignore_index=0,
                 loss_weight=1.0,
                 **kwards):

        self.loss_weight = loss_weight
        super(GDL_CE_loss, self).__init__({'batch_dice': batch_dice, 'smooth': dice_smooth, 'do_bg': do_bg, 'square_volumes': square_volumes}, 
                                          {'weight': class_weight, 'ignore_index': ignore_index},)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(GDL_CE_loss, self).forward(seg_logit, seg_label)

class GDL_Topk_loss(GDL_and_topk_loss):
    """GDL_Topk_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 dice_smooth=1.,
                 square_volumes=False,
                 class_weight=None,
                 ignore_index=255,
                 k=10,
                 loss_weight=1.0,
                 **kwards):

        self.loss_weight = loss_weight
        super(GDL_Topk_loss, self).__init__({'batch_dice': batch_dice, 'smooth': dice_smooth, 'do_bg': do_bg, 'square_volumes': square_volumes}, 
                                            {'weight': class_weight, 'ignore_index': ignore_index, 'k': k},)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(GDL_Topk_loss, self).forward(seg_logit, seg_label)


class DC_FC_CE(DC_FC_CE_loss):
    """DC_CE_loss.
    Rearrange code for callable
    """

    def __init__(self,
                 batch_dice=False, 
                 do_bg=True, 
                 smooth=1.,
                 class_weight=None, 
                 ignore_index=255, 
                 square_dice=False,
                 weight_ce=1, 
                 weight_dice=1,
                 weight_fc=1, 
                 log_dice=False, 
                 loss_weight=1.0,
                 
                 alpha=None, 
                 gamma=2, 
                 balance_index=0, 
                 focal_smooth=1e-5, 
                 
                 **kwards):

        self.loss_weight = loss_weight
        super(DC_FC_CE, self).__init__({'batch_dice': batch_dice, 'smooth': smooth, 'do_bg': do_bg}, 
                                         {'weight': class_weight},
                                         {'alpha': alpha, 'gamma': gamma, 'balance_index': balance_index, 'smooth': focal_smooth}, 
                                         weight_ce=weight_ce, 
                                         weight_dice=weight_dice,
                                         weight_fc=weight_fc, 
                                         log_dice=log_dice,
                                         ignore_label=ignore_index, 
                                         square_dice=square_dice)

    def forward(self, seg_logit, seg_label, **kwards):
        return self.loss_weight * super(DC_FC_CE, self).forward(seg_logit, seg_label)

if __name__ == '__main__':
    import torch
    a = torch.rand(2, 3, 16, 16) # out
    b = torch.randint(0, 2, (2, 1, 16, 16)) # gt
    
    a = torch.rand(1, 3, 5, 5)
    b = torch.argmax(a, 1).unsqueeze(1)
    
    # dcce = DC_CE_loss()
    # loss = dcce(a, b)
    # print(loss)
    
    # dcfc = DC_Focal_loss()
    # loss = dcfc(a, b)
    # print(loss)
    
    # gdlce = GDL_CE_loss()
    # loss = gdlce(a, b)
    # print(loss)

    dcfcce = DC_FC_CE()
    loss = dcfcce(a, b)
    print(loss)
    
    