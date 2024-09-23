#!/usr/bin/python3

import torch
from torch import nn

class WeightModel(nn.Module):
  def __init__(self, neighbor_num = 4):
    self.dense1 = nn.Linear(neighbor_num + 1, neighbor_num)
    self.gelu1 = nn.GELU()
    self.dense2 = nn.Linear(739, 1)
  def forward(self, x):
    # x.shape = (batch, neighbor_num + 1, 739)
    results = torch.permute(x, (0,2,1)) # x.shape = (batch, 739, neighbor_num + 1)
    results = self.dense1(results) # x.shape = (batch, 739, neighbor_num)
    results = self.gelu1(results)
    results = torch.permute(results, (0,2,1)) # x.shape = (batch, neighbor_num, 739)
    results = torch.squeeze(self.dense2(results), dim = -1) # x.shape = (batch, neighbor_num)
    results = torch.softmax(results, dim = -1) # results.shape = (batch, neighbor_num)
    return results

