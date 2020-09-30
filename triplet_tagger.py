## Class to define final, fully-connected layer
# [input] : [40,1] - softmax outputs of all 20 lorentzNN triplets
# [output] : [21,1] - raw logit output tagging correct triplet (or none)
# Author: Bobby Schiller
# Last Modified: 30 September 2020

import numpy as np
import torch
import lorentzNN as lNN

class triplet_tagger(torch.nn.Module):
  
  def __init__(self, **kwargs):
    super(triplet_tagger, self).__init__()
    self.lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/p_standard1.npz",i_shape=(200,4,3),device=torch.device('cuda')).to(device=torch.device('cuda'))
    self.layer1 = torch.nn.Linear(40,40)
    self.activation1 = torch.nn.ReLU()
    self.layer2 = torch.nn.Linear(40,21)
  
  def forward(self, x, **kwargs):
    # LorentzNN uses batch size of 200, so shaping input batches to fit
    N = len(x)
    lorentz_batches = int(N/10)
    x = torch.reshape(x,(lorentz_batches,10,20,4,3))
    x1 = torch.zeros((lorentz_batches,200,2))

    # Run triplet batches through LorentzNN
    for i in range(lorentz_batches):
      x1[i] = (self.lorentz_model((x[i].flatten(end_dim=1)))).to(device=torch.device('cuda'))

    # Reshape back to input size for triplet_tagger
    x = torch.reshape(x1,(N,40)).to(device=torch.device('cuda'),dtype=torch.double)
    x = self.layer1(x)
    x = self.activation1(x)
    x = self.layer2(x)

    # Cross entropy loss demands raw logits, so no activation
    return x
