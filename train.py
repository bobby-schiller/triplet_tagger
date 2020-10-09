## Code to train the fully-connected layer after the lorentzNN
# Author: Bobby Schiller
# Last Modified: 4 October 2020

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
event_num = 200000
batch_size = 100
max_epoch = 40

train_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_train.npz')
validation_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_validation.npz')
testing_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_testing.npz')

def train():

  parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
  parser.add_argument('-N','--num-epochs',
                        default=10, type=int,
                        help='Number of epochs')
  parser.add_argument('-L','--trained_layers',
                        default=False,
                        help='Pre-trained LorentzNN file')

  args = parser.parse_args()

  train_dataset = lorentz_format(train_data,event_num)
  validation_dataset = lorentz_format(validation_data,10000)
  testing_dataset = lorentz_format(testing_data,10000)

  # initialize triplet_tagger layer
  model = tt.triplet_tagger()
  model.double()
  model.to(device=torch.device('cuda')) 
  loss_array = []
  loss_func = torch.nn.CrossEntropyLoss()
  #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

  # lock gradients and load pre-trained lorentzNN layers, if desired
  if args.trained_layers:
    model.lorentz_model.requires_grad = False
    checkpoint = torch.load(args.trained_layers)
    model.lorentz_model.load_state_dict(checkpoint['model_state_dict'])
    model.lorentz_model.cola.Cij = checkpoint['cola_weights']
    model.lorentz_model.lola.w = checkpoint['lola_weights']
    model.lorentz_model.standardize.means_mat = checkpoint['standard_means']
    model.lorentz_model.standardize.stds_mat = checkpoint['standard_stds']

  evaluate(model,validation_dataset,"initial")

  ######## TRAINING ########
  current_min_loss = 10000000
  current_min_loss_epoch = 0
  epoch_loss = []
  for epoch in range(max_epoch):
    model.train()
    running_epoch_loss = 0
    for batch in train_dataset:
      out = model(batch[0].to(device=torch.device('cuda')))
      loss = loss_func(out,torch.squeeze((batch[1].long()).to(device=torch.device('cuda'))))
      running_epoch_loss += float(loss)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    ######## VALIDATION ########
    epoch_loss.append(running_epoch_loss)
    evaluate(model,validation_dataset,"e"+str(epoch))
    if running_epoch_loss < current_min_loss:
      current_min_loss = running_epoch_loss
      current_min_loss_epoch = epoch
    elif running_epoch_loss >= current_min_loss and (epoch-current_min_loss_epoch)>= 5:
      break

  ######## TESTING ########
  evaluate(model,testing_dataset,"final")

  plt.plot(epoch_loss)
  plt.savefig('/scratch365/rschill1/logs/loss_plot.png')
  plt.clf()

# Shape the data into the correct format for passing to lorentzNN
def lorentz_format(data,event_num):
  i,j = (0,0)
  jetv = np.zeros((event_num,20,4,3))
  for ev in data['input'][:event_num]:
    
    # Generate all 20 triplet combinations
    for iter in comb:
      # [particles,features]=>[features,particles]
      temp = np.take(ev,iter,0).transpose()
      # [px,py,pz,E]=>[E,px,py,pz] using row swaps
      temp[[0,3]] = temp[[3,0]]
      temp[[2,3]] = temp[[3,2]]
      temp[[1,2]] = temp[[2,1]]
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

    ######## ACCURACY AND PLOTTING ########
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
