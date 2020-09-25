## Parser to convert jets.npz file into correct input files
# Author: Bobby Schiller
# Last Modified: 20 September 2020

import numpy as np
from numpy import load
from itertools import combinations

print(np.random.permutation(20))

data = load('jets.npz')

#num_ev = len(data['match_validation'])
num_ev = 500000

# Generate combinations of triplets
comb = list(combinations(range(6),3))

# target array
match_train = np.zeros((num_ev,1))

# discard events with more than 3 matches (not triplets)
drop_event = []

drop = 0
hit = 0
miss = 0

# compute match for each triplet; append to new match_
for ev_i, event in enumerate(data['match_train']):
  index = hit + miss
  if np.sum(event) > 3:
    drop_event.append(ev_i)
    drop += 1
    continue
  elif np.sum(event) < 3:
    if miss == 0.1*num_ev:
      continue
    match_train[index] = 20
    miss += 1
    continue
  for iter_i, iter in enumerate(comb):
    if np.sum(np.take(event,iter,0)) == 3:
      if hit == 0.9*num_ev:
        break
      match_train[index] = iter_i
      hit += 1
      continue

target = match_train[:-(drop)]
input = np.delete(data['jetv_train'],drop_event,0)

# shuffle the data
shuffler = np.random.permutation(len(target))
target = target[shuffler]
input = input[shuffler]

np.savez("new_jt_trainFull",targets=target,input=input)
