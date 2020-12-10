## Code to train the triplet_tagger network
# Author: Bobby Schiller
# Last Modified: 20 October 2020

import argparse
import time
import numpy as np
from numpy import load
from itertools import combinations
import lorentzNN as lNN
import torch
import triplet_tagger1 as tt
import matplotlib.pyplot as plt
import torch.utils.data as dutils
import math
import jet_parser as jtp

######## SETUP ########

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = list(combinations(range(6),3))

learning_rate = 0.001
event_num = 300000
batch_size = 100
max_epoch = 40

validation_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_validation.npz')
testing_data = load('/scratch365/rschill1/nn/triplet_tagger/jt_testing.npz')

loss_func = torch.nn.CrossEntropyLoss()

def train():

  parser = argparse.ArgumentParser(description='Train ANN auto-encoder.')
  parser.add_argument('-N','--num-epochs',
                        default=10, type=int,
                        help='Number of epochs')
  parser.add_argument('-L','--trained_layers',
                        default=False,
                        help='Pre-trained LorentzNN file')
  parser.add_argument('-B','--batch_size',
                        default=100, type=int,
                        help='Batch Size')
  parser.add_argument('-S','--size',nargs=3,
                        default=[1000,500,100],
                        help='Space-delimited string containing layer sizes')
  parser.add_argument('-F','--filename')
  parser.add_argument('-W','--weighting',nargs=3,
                        default=[1,1,1],
                        help='Training data weightings')

  args = parser.parse_args()

  weighting = [float(i) for i in args.weighting]
  
  train_data = jtp.parse(weighting)

  size = [int(i) for i in args.size]

  train_dataset = lorentz_format(train_data,event_num)
  validation_dataset = lorentz_format(validation_data,10000)
  testing_dataset = lorentz_format(testing_data,10000)

  # initialize triplet_tagger layer
  model = tt.triplet_tagger(size)
  model.double()
  model.to(device=torch.device('cuda')) 
  #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

  # lock gradients and load pre-trained lorentzNN layers, if desired
  if args.trained_layers:
    #model.lorentz_model.requires_grad = False
    checkpoint = torch.load(args.trained_layers)
    model.lorentz_model.load_state_dict(checkpoint['model_state_dict'])
    model.lorentz_model.cola.Cij = checkpoint['cola_weights']
    model.lorentz_model.lola.w = checkpoint['lola_weights']
    model.lorentz_model.standardize.means_mat = checkpoint['standard_means']
    model.lorentz_model.standardize.stds_mat = checkpoint['standard_stds']

  evaluate(model,validation_dataset,"initial"+args.filename)

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
    current_epoch_loss = evaluate(model,validation_dataset,"e"+str(epoch))
    epoch_loss.append(current_epoch_loss)

    # Terminate training after 5 consecutive epochs with no improvement
    if current_epoch_loss < current_min_loss:
      current_min_loss = current_epoch_loss
      current_min_loss_epoch = epoch
      torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cola_weights': model.lorentz_model.cola.Cij,
                'lola_weights': model.lorentz_model.lola.w,
                'standard_means': model.lorentz_model.standardize.means_mat,
                'standard_stds': model.lorentz_model.standardize.stds_mat,
            }, ('/scratch365/rschill1/logs/best_{}.zip').format(args.filename))

    elif current_epoch_loss >= current_min_loss and (epoch - current_min_loss_epoch)>= 5:
      break

  ######## TESTING ########
  evaluate(model,testing_dataset,"final")

  plt.plot(epoch_loss)
  plt.savefig(('/scratch365/rschill1/logs/loss_plot_{}.png').format(args.filename))
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

    model.eval()
    # Get the output in batches
    total_out = []
    target = []
    loss = 0
    acc_count = 0
    for batch in dataset:
      out = model(batch[0].to(device=torch.device('cuda')))
      loss += float(loss_func(out,torch.squeeze((batch[1].long()).to(device=torch.device('cuda')))))
      total_out.append(out)
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

    ######## ACCURACY AND PLOTTING ########
    print(("Accuracy {}: {}").format(filename,str(acc_count/(len(out_array)))))
    out_array.sort()
    target.sort()
    plt.hist([out_array,target],bins=21)
    plt.savefig('/scratch365/rschill1/logs/svb_{}_{}.png'.format(filename,args.filename))
    plt.clf()

    correct_out_array.sort()
    plt.hist(correct_out_array,bins=21)
    plt.savefig('/scratch365/rschill1/logs/correct_out.png')
    plt.clf()"""
    print(loss)
    return loss

if __name__ == "__main__":
    train()
