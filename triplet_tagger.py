## Class to define final, fully-connected layer
# [input] : [40,1] - softmax outputs of all 20 lorentzNN triplets
# [output] : [21,1] - softmax output tagging correct triplet (or none)
# Author: Bobby Schiller
# Last Modified: 9 September 2020

import numpy as np
import torch
import lorentzNN as lNN

class triplet_tagger(torch.nn.Module):
  
  def __init__(self, **kwargs):
    super(triplet_tagger, self).__init__()
    self.lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/p_standard1.npz",i_shape=(200,4,3),device=torch.device('cuda')).to(device=torch.device('cuda'))
    self.layer1 = torch.nn.Linear(40,21)
    self.activation1 = torch.nn.LeakyReLU()
    self.layer2 = torch.nn.Linear(21,21)
  
  def forward(self, x, **kwargs):
    # Unpack the events from each batch and run lorentzNN
    x = torch.flatten(x,end_dim=1)
    x = (self.lorentz_model(x)).to(device=torch.device('cuda'))
    
    # Reshape back to input size for triplet_tagger
    x = torch.reshape(x,(10,40))
    x = self.layer1(x)
    x = self.activation1(x)
    x = self.layer2(x)

    # Cross entropy loss demands raw logits, so no activation
    return x
