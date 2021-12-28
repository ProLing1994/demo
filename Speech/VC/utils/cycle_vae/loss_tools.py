import torch
from torch import nn


class MCDloss(nn.Module):
    """ spectral loss based on mel-cepstrum distortion (MCD) """
    def __init__(self):
        super(MCDloss, self).__init__()
        self.frac10ln2 = (10.0/2.3025850929940456840179914546844)
        self.sqrt2 = 1.4142135623730950488016887242097
    
    def forward(self, x, y, twf=None, L2=False):
        """
            twf is time-warping function, none means exact same time-alignment
            L2 means using squared loss (L2-based loss), false means using abs./L1-based loss; default false
        """
        if twf is None:
            if not L2:
                mcd = self.frac10ln2*self.sqrt2*torch.sum(torch.abs(x-y),1)
            else:
                mcd = self.frac10ln2*torch.sqrt(2.0*torch.sum((x-y).pow(2),1))
        else:
            if not L2:
                mcd = self.frac10ln2*self.sqrt2*torch.sum(torch.abs(torch.index_select(x,0,twf)-y),1)
            else:
                mcd = self.frac10ln2*torch.sqrt(2.0*torch.sum((torch.index_select(x,0,twf)-y).pow(2),1))
        mcd_sum = torch.sum(mcd)
        mcd_mean = torch.mean(mcd)
        mcd_std = torch.std(mcd)
        return mcd_sum, mcd_mean, mcd_std
    

def loss_vae_laplace(param, clip=False, lat_dim=None):
    if lat_dim is None:
        lat_dim = int(param.shape[1]/2)
    mu = param[:, :lat_dim]
    sigma = param[:, lat_dim:]
    #if clip and torch.min(sigma) < -10.708206508753178232789577606809: #1e-9
    if clip and torch.min(sigma) < -14.162084148244246758816564788835: #1e-12
        #sigma = torch.clamp(sigma,min=-7.2543288692621097067625904247823) #1e-6
        #sigma = torch.clamp(sigma,min=-10.708206508753178232789577606809) #1e-9
        sigma = torch.clamp(sigma,min=-14.162084148244246758816564788835) #1e-12
    mu_abs = mu.abs()
    scale = torch.exp(sigma)
    return torch.mean(torch.sum(-sigma+scale*torch.exp(-mu_abs/scale)+mu_abs-1,1)) # log_scale