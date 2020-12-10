## Parser to convert jets.npz file into correct input files
# Author: Bobby Schiller
# Last Modified: 30 October 2020

import numpy as np
from numpy import load
from itertools import combinations

data = load('/scratch365/rschill1/nn/triplet_tagger/jets.npz')
#arr = ["train","validation","testing"]
arr = ["validation","testing"]

# Generate combinations of triplets
comb = list(combinations(range(6),3))

balance = [1,1,1]
## balance := [misses,hits,individual_hits]
# for unbalanced data, use [1,1,1]
for arr_name in arr:
  
  num_ev = int(len(data[("match_"+arr_name)]))

  hit = 0
  miss = 0
  hits = np.zeros(20)
  total = 0
  events = []
  matches = []
 
  # compute match for each triplet; append to new match_
  for ev_i, event in enumerate(data[("match_"+arr_name)]):
    total = ev_i + 1
    index = hit + miss

    # skip over events with more than 3 matches (not triplets)
    if np.sum(event) > 3:
      continue
    elif np.sum(event) < 3:
      if miss+1 >= balance[0]*num_ev:
        continue
      miss += 1
      matches.append(20)
      events.append(ev_i)
      continue
    for iter_i, iter in enumerate(comb):
      if np.sum(np.take(event,iter,0)) == 3:
        if hits[iter_i]+1 > (balance[2])*num_ev:
          if hit+1 >= (balance[1])*num_ev:
            break
          continue
        matches.append(iter_i)
        events.append(ev_i)
        hit += 1
        hits[iter_i] += 1
        continue

  target = np.array(matches)
  input = np.take(data['jetv_'+arr_name],events,0)

  # shuffle the data
  shuffler = np.random.permutation(len(target))
  target = target[shuffler]
  input = input[shuffler]
  print(len(target))
  print(len(input))
  np.savez('jt_'+arr_name,targets=target,input=input)
