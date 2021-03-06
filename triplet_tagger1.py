## Class to define final, fully-connected layer
# [input] : [20,4,3] - all 20 triplets in an event
# [output] : [21] - raw logit output tagging correct triplet (or none)
# Author: Bobby Schiller
# Last Modified: 29 October 2020

import numpy as np
import torch
import lorentzNN as lNN

class triplet_tagger(torch.nn.Module):
  
  def __init__(self, size = [1000,500,100], **kwargs):
    super(triplet_tagger, self).__init__()
    self.lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/p_standard1.npz",i_shape=(200,4,3),device=torch.device('cuda')).to(device=torch.device('cuda'))
    self.layer1 = torch.nn.Linear(500,size[0])
    torch.nn.init.kaiming_normal_(self.layer1.weight,mode='fan_in')
    torch.nn.init.constant_(self.layer1.bias,0)
    self.activation1 = torch.nn.ReLU()
    self.layer2 = torch.nn.Linear(size[0],size[1])
    torch.nn.init.kaiming_normal_(self.layer2.weight,mode='fan_in')
    torch.nn.init.constant_(self.layer2.bias,0)
    self.activation2 = torch.nn.ReLU()
    self.layer3 = torch.nn.Linear(size[1],size[2])
    torch.nn.init.kaiming_normal_(self.layer3.weight,mode='fan_in')
    torch.nn.init.constant_(self.layer3.bias,0)
    self.activation3 = torch.nn.ReLU()
    self.layer4 = torch.nn.Linear(size[2],21)
    torch.nn.init.kaiming_normal_(self.layer3.weight,mode='fan_in')
    torch.nn.init.constant_(self.layer4.bias,0)  

  def forward(self, x, **kwargs):
    # LorentzNN uses batch size of 200, so shaping input batches to fit
    N = len(x)
    lorentz_batches = int(N/10)
    x = torch.reshape(x,(lorentz_batches,10,20,4,3))
    x1 = torch.zeros((lorentz_batches,10,500))

    # Run triplet batches through LorentzNN
    for i in range(lorentz_batches):
      temp = (self.lorentz_model((x[i].flatten(end_dim=1)))).to(device=torch.device('cuda'))
      temp = torch.reshape(temp,(10,500))
      for j, score in enumerate(temp):
        x1[i][j] = score

    # Reshape back to input size for triplet_tagger
    x = torch.reshape(x1,(N,500)).to(device=torch.device('cuda'),dtype=torch.double)
    x = self.layer1(x)
    x = self.activation1(x)
    x = self.layer2(x)
    x = self.activation2(x)
    x = self.layer3(x)
    x = self.activation3(x)
    x = self.layer4(x)

    # Cross entropy loss demands raw logits, so no activation
    return x
