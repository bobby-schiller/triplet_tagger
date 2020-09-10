## Class to define final, fully-connected layer
# [input] : [40,1] - softmax outputs of all 20 lorentzNN triplets
# [output] : [21,1] - softmax output tagging correct triplet (or none)
# Author: Bobby Schiller
# Last Modified: 9 September 2020

import numpy as np
import torch

class triplet_tagger(torch.nn.Module):
  
  def __init__(self, **kwargs):
    super(triplet_tagger, self).__init__()
    self.layer = torch.nn.Linear(40,21)
    self.activation = torch.nn.Softmax(dim=1)
  
  def forward(self, x, **kwargs):
    x = self.layer(x)
    x = self.activation(x)
    return x
