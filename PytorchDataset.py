import torch
import numpy
import string

from torch.utils import data
import python_speech_features as sf

class PytorchDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'

        X = torch.tensor(self.list_IDs[index])
        y = torch.tensor(self.labels[index])

        return X, y