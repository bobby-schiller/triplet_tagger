## Code to train the fully-connected layer after the lorentzNN
# Author: Bobby Schiller
# Last Modified: 3 September 2020

import argparse
import time
import numpy as np
from numpy import load
from itertools import combinations
import lorentzNN.lorentzNN as lNN
import lorentzNN.model_handler as mh
import lorentzNN.model_benchmark as bench
import torch

model_file = 'out_file_e_20.tar'

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = combination(range(6),3)

learning_rate = 0.001

def train():
  
  # initialize LorentzNN model
  lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/inputs/ \
                      lNN_standards/p_standard1.npz",
                      i_shape=input_shape,
                      device=args.device).to(device=args.device)
  
  # Define optimizer and loss functions
  loss_func = torch.nn.BCELoss()
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # Now build ModelHandler and load pretrained LorentzNN
  handler = mh.ModelHandler(lorentz_model, loss_func, optim, batch_size=1)
  handler.loadModel(model_file, True)
  
  # Run through the data and calculate triplet probabilities
  lorentz_out = handler.model()
  
  # initialize triplet_tagger layer
  triplet_model = triplet_tagger()
  
  # Train the triplet_tagger layer
  max_epoch = 20
  for epoch in range(max_epoch):
    epoch_loss = 0
    
    for minibatch in range(num_minibatch):
      out = model(inputMiniBatches[miniBatch])
      loss = loss_func(out,outputMiniBatches[miniBatch])
      loss_epoch += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
