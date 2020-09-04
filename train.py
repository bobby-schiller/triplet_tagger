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

# initialize model
model = lNN.LorentzNN("/scratch365/rschill1/nn/inputs/ \
                      lNN_standards/p_standard1.npz",
                      i_shape=input_shape,
                      device=args.device).to(device=args.device)

# generate list of combinations of 3 jets in [0,5]
# itertools provides standard indexing for the triplet combinations 
comb = combination(range(6),3)

for i in range(20):
  model.deepcopy()
