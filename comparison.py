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
import triplet_tagger1 as tt1
import math

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = list(combinations(range(6),3))

comb2 = list(combinations(range(1,9),2))
sizes = [[1000,500,100],[1000,500,100],[1000,500,100],[1000,500,100],[2000,1000,100],[2000,1000,100],[2000,1000,100],[2000,1000,100],[2000,1000,100]]

learning_rate = 0.001
event_num = 300000
batch_size = 100

testing_data = load('/scratch365/rschill1/nn/triplet_tagger/un_jt_testing.npz')


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
  lorentz_dataset = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
  return lorentz_dataset

def evaluate(model,model2,dataset,filename):
  with torch.no_grad():
    
    # Get the output in batches
    total_out = []
    total_out1 = []
    target = []
    acc_count = 0
    for batch in dataset:
      out = model(batch[0].to(device=torch.device('cuda')))
      total_out.append(out)
      out = model2(batch[0].to(device=torch.device('cuda')))
      total_out1.append(out)
      for match in batch[1]:
        target.append(float(match.item()))

    i = 0
    out_array = []
    correct_out_array = []
    for batch in total_out:
      for event in batch:
        hit = float(torch.argmax(event).item())
        out_array.append(hit)
        if hit == target[i]:
          acc_count+=1
          correct_out_array.append(target[i])
        i += 1

    i = 0
    out_array2 = []
    for batch in total_out1:
      for event in batch:
        hit = float(torch.argmax(event).item())
        out_array2.append(hit)
        i += 1

    target = [i+0.5 for i in target]
    out_array = [i+0.5 for i in out_array]
    out_array2 = [i+0.5 for i in out_array2]

    targ1 = target.copy()
    targ2 = target.copy()
    out_targ = out_array.copy()
    out2_targ = out_array2.copy()
    comp_out = out_array.copy()
    comp_out2 = out_array2.copy()

    print(len(targ1))
    print(len(targ2))
    print(len(out_targ))
    print(len(out2_targ))
    print(len(comp_out))
    print(len(comp_out2))

    ind1 = []
    ind2 = []
    ind3 = []

    for i in range(len(out_array2)):
      if target[i] == out_array[i]:
        ind1.append(i)
      if target[i] == out_array2[i]:
        ind2.append(i)
      if out_array[i] == out_array2[i]:
        ind3.append(i)

    targ1 = np.delete(targ1,ind1)
    out_targ = np.delete(out_targ,ind1)
    targ2 = np.delete(targ2,ind2)
    out2_targ = np.delete(out2_targ,ind2)
    comp_out = np.delete(comp_out,ind3)
    comp_out2 = np.delete(comp_out2,ind3)

    edges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    counts,binsx,binsy,n=plt.hist2d(comp_out,comp_out2,bins=edges,density=True)
    """for i in range(21):
      counts[i]"""
    plt.colorbar()
    plt.xlabel(("Network {}").format(filename[1]))
    plt.ylabel(("Network {}").format(filename[0]))
    plt.title(("Network {} vs. {}").format(filename[0],filename[1]))
    plt.savefig(('/scratch365/rschill1/logs/hist{}_{}.png').format(filename[0],filename[1]))
    plt.clf()

    edges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    counts,binsx,binsy,n=plt.hist2d(targ1,out_targ,bins=edges,density=True)
    for i in range(21):
      counts[i][i] = None
    plt.colorbar()
    plt.xlabel("Target")
    plt.ylabel(("Network {}").format(filename[0]))
    plt.title(("Network {} vs. Target").format(filename[0]))
    plt.savefig(('/scratch365/rschill1/logs/hist{}.png').format(filename[0]))
    plt.clf()

    edges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    counts,binsx,binsy,n=plt.hist2d(targ2,out2_targ,bins=edges,density=True)
    plt.colorbar()
    plt.xlabel("Target")
    plt.ylabel(("Network {}").format(filename[1]))
    plt.title(("Network {} vs. Target").format(filename[1]))
    plt.savefig(('/scratch365/rschill1/logs/hist{}.png').format(filename[1]))
    plt.clf()

for pair in comb2:

  parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
  parser.add_argument('-L','--trained_layers',
                        default=False,
                        help='Pre-trained LorentzNN file')
  args = parser.parse_args()
  testing_dataset = lorentz_format(testing_data,10000)

  # initialize triplet_tagger layer
  model1 = tt1.triplet_tagger(sizes[pair[0]-1])
  model1.double()
  model1.to(device=torch.device('cuda'))
  if args.trained_layers:
    model1.requires_grad = False
    checkpoint = torch.load(("/scratch365/rschill1/logs/best_{}.zip").format(str(pair[0])))
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.lorentz_model.cola.Cij = checkpoint['cola_weights']
    model1.lorentz_model.lola.w = checkpoint['lola_weights']
    model1.lorentz_model.standardize.means_mat = checkpoint['standard_means']
    model1.lorentz_model.standardize.stds_mat = checkpoint['standard_stds']

  model2 = tt1.triplet_tagger(sizes[pair[1]-1])
  model2.to(device=torch.device('cuda'))
  model2.double()
  if args.trained_layers:
    model2.requires_grad = False
    checkpoint = torch.load(("/scratch365/rschill1/logs/best_{}.zip").format(str(pair[1])))
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.lorentz_model.cola.Cij = checkpoint['cola_weights']
    model2.lorentz_model.lola.w = checkpoint['lola_weights']
    model2.lorentz_model.standardize.means_mat = checkpoint['standard_means']
    model2.lorentz_model.standardize.stds_mat = checkpoint['standard_stds']

  ######## TESTING ########
  evaluate(model1,model2,testing_dataset,pair)

