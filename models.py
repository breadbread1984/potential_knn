#!/usr/bin/python3

import torch
from torch import nn

def WeightModel(neighbor_num = 4):
  return nn.Sequential(nn.Linear(1 + neighbor_num, neighbor_num),
                       nn.ReLU(),
                       nn.Linear(neighbor_num, neighbor_num),
                       nn.Softmax())
