## Code to train the fully-connected layer after the lorentzNN
# Author: Bobby Schiller
# Last Modified: 17 September 2020

import argparse
import time
import numpy as np
from numpy import load
from itertools import combinations
import lorentzNN.lorentzNN as lNN
import lorentzNN.model_handler as mh
import lorentzNN.model_benchmark as bench
import torch
import matplotlib.pyplot as plt

model_file = 'out_file_e_20.tar'

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = list(combination(range(6),3))

learning_rate = 0.001

data = load('new_jt_trainFull.npz')
data_cat = load('new_jt_trainFull_cat.npz')
np_train = data['input']
trip_targ_cat = data_cat['targets']
trip_targ = data['targets']

def train():
  # initialize LorentzNN model
  lorentz_model = lNN.LorentzNN("/scratch365/rschill1/nn/inputs/ \
                      lNN_standards/p_standard1.npz",
                      i_shape=(200,4,3))
  
  # Define optimizer and loss functions
  loss_func = torch.nn.BCELoss()
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # Now build ModelHandler and load pretrained LorentzNN
  handler = mh.ModelHandler(lorentz_model, loss_func, optim, batch_size=200)
  handler.loadModel(model_file, True)
  
  # Generate the dataset in the form required by LorentzNN, with dummy targets
  i = 0
  triplets = np.array((10000000,4,3))
  for ev in np_train[:500000]:
    for iter_i, iter in comb:
      triplets[i] = np.take(np_train,iter).transpose()
      i+=1
  
  inp = torch.as_tensor(triplets, dtype=torch.double)
  targets = torch.as_tensor(np.zeros(10000000), dtype=torch.double)
  dataset = dutils.TensorDataset(inp, targets)
  dutils.DataLoader(dataset, batch_size=200, shuffle=False, drop_last=False)
  
  # Run through the data and calculate triplet probabilities
  lorentz_out = handler.model()
  
  # Collect the output data and organize it into events
  trip_in = np.zeros((500000,40))
  i = 0
  j = 0
  for trip in lorentz_out:
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
  if trainSize % args.batch_size != 0:
    print ('Warning: Training set size ({}) does not divide evenly into batches of {}'.format(trainSize,args.batch_size),file=out_stream)
    print ('-->Discarding the remaider, {} examples'.format(trainSize % args.batch_size),file=out_stream)
    numMiniBatch -= 1
  
  # initialize triplet_tagger layer
  triplet_model = triplet_tagger()
  
  loss_array = []
  loss_func1 = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(triplet_model.parameters(), lr=learning_rate)
  
  # Train the triplet_tagger layer
  max_epoch = 20
  for epoch in range(max_epoch):
    epoch_loss = 0
    
    for minibatch in range(num_minibatch):
      out = model(inputMiniBatches[miniBatch])
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
