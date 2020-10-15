import os

from torch.utils.data import Dataset

class SpeechDataset(Dataset):
  """
  Training dataset for Key word spotting
  """
  def __init__(self, cfg):

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.image_list)

  def __getitem__(self, index):
    """ get the item """