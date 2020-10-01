## Code to train the fully-connected layer after the lorentzNN
# Author: Bobby Schiller
# Last Modified: 20 September 2020

# TODO: Refactor and clean up variable names

import argparse
import time
import numpy as np
from numpy import load
from itertools import combinations
import lorentzNN as lNN
import torch
import triplet_tagger as tt
import matplotlib.pyplot as plt
import torch.utils.data as dutils
import math

trained_layers = '/scratch365/rschill1/nn/lorentzNN/run02e20.tar'
#trained_layers = False

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = list(combinations(range(6),3))

learning_rate = 0.001
event_num = 400000
batch_size = 100

train_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_trainFull_balanced.npz')
validation = load('/scratch365/rschill1/nn/triplet_tagger/new_new_jt_testingFull.npz')
#testing = load('/scratch365/rschill1/nn/triplet_tagger/jt_testingFull.npz')

def train():

  lorentz_dataset = lorentz_format(train_data,event_num)

  # initialize triplet_tagger layer
  model = tt.triplet_tagger()
  model.double()
  model.to(device=torch.device('cuda')) 
  loss_array = []
  loss_func1 = torch.nn.CrossEntropyLoss()
  #optimizer = torch.optim.Adam(triplet_model.parameters(),lr=learning_rate)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
  validation_dataset = lorentz_format(validation,10000)
  evaluate(model,validation_dataset,"initial")

  # lock gradients and load pre-trained lorentzNN layers, if desired
  if trained_layers:
    model.lorentz_model.requires_grad = False
    checkpoint = torch.load(trained_layers)
    model.lorentz_model.load_state_dict(checkpoint['model_state_dict'])
    model.lorentz_model.cola.Cij = checkpoint['cola_weights']
    model.lorentz_model.lola.w = checkpoint['lola_weights']
    model.lorentz_model.standardize.means_mat = checkpoint['standard_means']
    model.lorentz_model.standardize.stds_mat = checkpoint['standard_stds']

  # Train the triplet_tagger layer
  max_epoch = 10
  epoch_loss = []
  for epoch in range(max_epoch):
    running_epoch_loss = 0
    for batch in lorentz_dataset:
      out = model(batch[0].to(device=torch.device('cuda')))
      loss = loss_func1(out,torch.squeeze((batch[1].long()).to(device=torch.device('cuda'))))
      #loss_array.append(float(loss))
      running_epoch_loss += float(loss)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    epoch_loss.append(running_epoch_loss)
    evaluate(model,validation_dataset,"e"+str(epoch))

  ######## VALIDATION ########
  
  evaluate(model,validation_dataset,"final")
  """
  plt.plot(loss_array)
  plt.savefig('/scratch365/rschill1/logs/loss_plot.png')
  plt.clf()"""

  plt.plot(epoch_loss)
  plt.savefig('/scratch365/rschill1/logs/epoch_loss_plot.png')
  plt.clf()

"""
  total_out = []
  for batch in validation_dataset:
    with torch.no_grad:
      total_out.append(model(batch[0].to(device=torch.device('cuda'))))
  out_array = []
  for batch in total_out:
    for i,event in enumerate(batch):
      hit = torch.argmax(event).item()
      out_array.append(hit)


  # evaluate without triplet_tagger
  jet_acc = 0
  for ev_i, event in enumerate(trip_in):
    hits = event[1::2]
    if max(hits) > 0.5:
      hit = np.argmax(hits)
    else:
      hit = 0
    if trip_targ[ev_i] == hit:
      jet_acc += 1

  acc = acc_count/(len(total_out*len(total_out[0])))
  print("ACC: {}".format(acc))
  
  acc1 = jet_acc/len(total_out*len(total_out[0]))
  print("ACC1: {}".format(acc1)) 

  
  ######## PLOTTING ########
  plt.hist(out_array)
  plt.savefig('/scratch365/rschill1/logs/out_plot.png')
  plt.clf()
""" 

# Shape the data into the correct format for passing to lorentzNN
def lorentz_format(data,event_num):
  i,j = (0,0)
  jetv = np.zeros((event_num,20,4,3))
  for ev in data['input'][:event_num]:
    
    # Generate all 20 triplet combinations
    for iter in comb:
      # [particles,features]=>[features,particles]
      temp = np.take(ev,iter,0).transpose()
      # [px,py,pz,E]=>[E,px,py,pz]
      temp[[0,3]] = temp[[3,0]]
      jetv[i][j] = temp
      j+=1
    j=0
    i+=1

  input = torch.as_tensor(jetv, dtype=torch.double)
  targets = torch.as_tensor(data['targets'][:event_num], dtype=torch.double)
  dataset = dutils.TensorDataset(input, targets)
  lorentz_dataset = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  return lorentz_dataset

def evaluate(model,dataset,filename):
  with torch.no_grad():

    # Get the output in batches
    total_out = []
    target = []
    acc_count = 0
    for batch in dataset:
      out = model(batch[0].to(device=torch.device('cuda')))
      total_out.append(out)
      for match in batch[1]:
        target.append(float(match.item()))

    out_array = []
    correct_out_array = []
    for batch in total_out:
      for i,event in enumerate(batch):
        hit = float(torch.argmax(event).item())
        out_array.append(hit)
        if hit == target[i]:
          acc_count+=1
          correct_out_array.append(target[i])
    print(acc_count/(len(out_array)))
    out_array.sort()
    target.sort()
    plt.hist([out_array,target],bins=21)
    plt.savefig('/scratch365/rschill1/logs/svb_{}.png'.format(filename))
    plt.clf()

    correct_out_array.sort()
    plt.hist(correct_out_array,bins=21)
    plt.savefig('/scratch365/rschill1/logs/correct_out.png')
    plt.clf()

if __name__ == "__main__":
    train()
