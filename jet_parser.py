## Parser to convert jets.npz file into correct input files
# Author: Bobby Schiller
# Last Modified: 20 September 2020

import numpy as np
from numpy import load
from itertools import combinations

data = load('jets.npz')

num_ev = len(data['match_validation'])

# Generate combinations of triplets
comb = list(combinations(range(6),3))

# categorical and numerical forms
match_train_cat = np.zeros((num_ev,21))
match_train = np.zeros((num_ev,1))

# discard events with more than 3 matches (not triplets)
drop_event = []

# compute match for each triplet; append to new match_
for ev_i, event in enumerate(data['match_validation']):
  if np.sum(event) > 3:
    drop_event.append(ev_i)
    continue
  elif np.sum(event) < 3:
    match_train_cat[ev_i][20] = 1
    match_train[ev_i] = 20
    continue
  for iter_i, iter in enumerate(comb):
    if np.sum(np.take(event,iter,0)) == 3:
      match_train_cat[ev_i][iter_i] = 1
      match_train[ev_i] = iter_i
      continue

target_cat = np.delete(match_train_cat,drop_event,0)
target = np.delete(match_train,drop_event,0)
input = np.delete(data['jetv_validation'],drop_event,0)

np.savez("new_jt_validationFull_cat",targets=target_cat,input=input)
np.savez("new_jt_validationFull",targets=target,input=input)
