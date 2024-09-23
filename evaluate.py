#!/usr/bin/python3

from absl import flags, app
from os import listdir
import faiss
import numpy as np
import torch
from torch import nn, device, load, autograd
from torch.utils.data import DataLoader
from models import WeightModel
from create_dataset import RhoDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset npy')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset npy')
  flags.DEFINE_integer('k', default = 4, help = 'nearest neighbor number')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_enum('dist', default = 'l2', enum_values = {'l2','cos'}, help = 'distance type')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def main(unused_argv):
  # 1) create KD-tree
  res = faiss.StandardGpuResources()
  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0
  if FLAGS.dist == 'l2':
    index = faiss.GpuIndexFlatL2(res, 739, flat_config)
  elif FLAGS.dist == 'cos':
    index = faiss.GpuIndexFlatIP(res, 739, flat_config)
  print('load trainset')
  trainset = list()
  for f in listdir(FLAGS.trainset):
    trainset.append(np.load(join(FLAGS.trainset, f)))
  trainset = np.concatenate(trainset, axis = 0)
  trainset_rho, trainset_label = np.ascontiguousarray(trainset[:,:739].astype(np.float32)), np.ascontiguousarray(trainset[:,769:].astype(np.float32))
  assert trainset_label.shape[1] == 5
  faiss.normalize_L2(trainset_rho)
  index.add(trainset_rho)
  # 2) evaluation
  evalset = RhoDataset(FLAGS.evalset)
  loader = DataLoader(evalset, batch_size = FLAGS.batch, shuffle = True)
  model = WeightModel(FLAGS.k).to('cuda')
  ckpt = load(FLAGS.ckpt)
  model.load_state_dict(ckpt['state_dict'])
  pred_exc_list, pred_vxc_list = list(), list()
  exc_list, vxc_list = list(), list()
  for rho, pos, exc, vxc in loader:
    faiss.normalize_L2(rho.cpu().numpy())
    D, I = index.search(rho, FLAGS.k) # D.shape = (batch, k) I.shape = (batch, k)
    neighbor = torch.from_numpy(trainset_rho[I,:]).to('cuda') # neighbor.shape = (batch, k, 739)
    neighbor_exc = torch.from_numpy(trainset_label[I,3:4]).to('cuda') # neighbor_exc.shape = (batch,k,1)
    neighbor_vxc = torch.from_numpy(trainset_label[I,4:5]).to('cuda') # neighbor_vxc.shape = (batch,k,1)
    rho = rho.to('cuda')
    exc = exc.to('cuda')
    vxc = vxc.to('cuda')
    rho.requires_grad = True
    x = torch.cat([torch.unsqueeze(rho, dim = 1), neighbor], dim = 1) # x.shape = (batch, k+1, 739)
    weights = model(x) # weights.shape = (batch, k, 1)
    pred_exc = torch.sum(weights * neighbor_exc, dim = (1,2)) # pred_exc.shape = (batch)
    pred_vxc = autograd.grad(torch.sum(rho[:,rho.shape[1]//2] * pred_exc), rho, create_graph = True)[0][:,rho.shape[1]//2] + pred_exc # pred_vxc.shape = (batch)
    pred_exc_list.append(pred_exc)
    pred_vxc_list.append(pred_vxc)
    exc_list.append(exc)
    vxc_list.append(vxc)
  pred_exc = torch.cat(pred_exc_list)
  pred_vxc = torch.cat(pred_vxc_list)
  exc = torch.cat(exc_list)
  vxc = torch.cat(vxc_list)
  mae = nn.L1Loss()
  exc_mae = mae(pred_exc, exc)
  vxc_mae = mae(pred_vxc, vxc)
  print(f"exc MAE: {exc_mae} vxc MAE: {vxc_mae}")

if __name__ == "__main__":
  add_options()
  app.run(main)
