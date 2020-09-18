## Code to train the fully-connected layer after the lorentzNN
# Author: Bobby Schiller
# Last Modified: 17 September 2020

import argparse
import time
import numpy as np
from numpy import load
from itertools import combinations
import lorentzNN as lNN
import model_handler as mh
import model_benchmark as bench
import torch
import triplet_tagger as tt
import matplotlib.pyplot as plt
import torch.utils.data as dutils
import math

model_file = '/scratch365/rschill1/nn/lorentzNN/run04e20.tar'

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = list(combinations(range(6),3))

learning_rate = 0.001

data = load('/scratch365/rschill1/nn/triplet_tagger/new_jt_trainFull.npz')
data_cat = load('/scratch365/rschill1/nn/triplet_tagger/new_jt_trainFull_cat.npz')
np_train = data['input']
trip_targ_cat = data_cat['targets']
trip_targ = data['targets']

def train():
  print(torch.cuda.is_available())
  # initialize LorentzNN model
  lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/p_standard1.npz",
                      i_shape=(200,4,3),device=torch.device('cuda')).to(device=torch.device('cuda'))
  
  # Define optimizer and loss functions
  loss_func = torch.nn.BCELoss()
  optim = torch.optim.Adam(lorentz_model.parameters(), lr=learning_rate)
  
  # Now build ModelHandler and load pretrained LorentzNN
  handler = mh.ModelHandler(lorentz_model, loss_func, optim, batch_size=200)
  handler.loadModel(model_file, True)
  
  # Generate the dataset in the form required by LorentzNN, with dummy targets
  i = 0
  triplets = np.zeros((1000000,4,3))
  for ev in np_train[:50000]:
    for iter in comb:
      temp = np.take(ev,iter,0).transpose()
      temp[[0,3]] = temp[[3,0]]
      triplets[i] = temp
      i+=1
  
  inp = torch.as_tensor(triplets, dtype=torch.double)
  targets = torch.as_tensor(np.ones(1000000), dtype=torch.double)
  dataset = dutils.TensorDataset(inp, targets)
  lorentz_dataset = dutils.DataLoader(dataset, batch_size=200, shuffle=False, drop_last=False)
  for i in lorentz_dataset:
    print(i[0])
    break 
  # Run through the data and calculate triplet probabilities
  lorentz_out = []
  with torch.no_grad():
    for batch in lorentz_dataset:
      handler.model.eval()
      lorentz_out.append(handler.model(torch.as_tensor(batch[0],device=torch.device('cuda'))))
   
  print(lorentz_out[0][0])
  # Collect the output data and organize it into events
  trip_in = np.zeros((50000,40))
  i = 0
  j = 0
  for batch in lorentz_out:
    for trip in batch:
      for match in trip:
        trip_in[i][j] = match
        j+=1
        if j == 40:
          i+=1
          j=0
  
  # Send the lorentz_in and targets to tensors
  triplet_input = torch.as_tensor(trip_in,dtype=torch.double)
  triplet_targets = torch.as_tensor(trip_targ_cat,dtype=torch.double)
  
  batch_size = 200
  
  # Break up the training tensors into mini-batches
  trainSize = triplet_input.shape[0]
  numMiniBatch = int(math.ceil(trainSize/float(batch_size)))
  inputMiniBatches = triplet_input.chunk(numMiniBatch)
  outputMiniBatches = triplet_targets.chunk(numMiniBatch)
  if trainSize % 200 != 0:
    print ('Warning: Training set size ({}) does not divide evenly into batches of {}'.format(trainSize,args.batch_size),file=out_stream)
    print ('-->Discarding the remaider, {} examples'.format(trainSize % args.batch_size),file=out_stream)
    numMiniBatch -= 1
  
  # initialize triplet_tagger layer
  triplet_model = tt.triplet_tagger()
  
  loss_array = []
  loss_func1 = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(triplet_model.parameters(), lr=learning_rate)
  
  # Train the triplet_tagger layer
  max_epoch = 20
  for epoch in range(max_epoch):
    epoch_loss = 0
    
    for minibatch in range(numMiniBatch):
      out = triplet_model(inputMiniBatches[miniBatch])
      loss = loss_func1(out,outputMiniBatches[miniBatch])
      epoch_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    loss_array.append(epoch_loss)
  
  # compute the accuracy
  acc_array = []
  
  epoch_array = np.arange(0,max_epoch)
  plt.plot(epoch_array,loss_array)
  plt.savefig('/scratch365/rschill1/nn/loss_plot.png')


if __name__ == "__main__":
    train()
