#!/usr/bin/python3

from os import listdir, mkdir
from os.path import join, exists, splitext
from absl import flags, app
import faiss
import numpy as np
import torch
from torch import nn, device, save, load, no_grad, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models import WeightModel
from create_dataset import RhoDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset npy')
  flags.DEFINE_string('evalset', default = None, help = 'path to evalset npy')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('k', default = 4, help = 'nearest neighbor number')
  flags.DEFINE_integer('batch', default = 1024, help = 'batch size')
  flags.DEFINE_enum('dist', default = 'l2', enum_values = {'l2','cos'}, help = 'distance type')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('epochs', default = 200, help = 'number of epochs')

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
  # 2) training
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  evalset = RhoDataset(FLAGS.evalset)
  loader = DataLoader(evalset, batch_size = FLAGS.batch, shuffle = True)
  model = WeightModel(FLAGS.k).to('cuda')
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  mae = nn.L1Loss()
  for epoch in range(FLAGS.epochs):
    for step, (rho, pos, exc, vxc) in enumerate(loader):
      # search for NN
      faiss.normalize_L2(rho.cpu().numpy())
      D, I = index.search(rho, FLAGS.k) # D.shape = (batch, k) I.shape = (batch, k)
      neighbor = torch.from_numpy(trainset_rho[I,:]).to('cuda') # neighbor.shape = (batch, k, 739)
      neighbor_exc = torch.from_numpy(trainset_label[I,3:4]).to('cuda') # neighbor_exc.shape = (batch,k,1)
      neighbor_vxc = torch.from_numpy(trainset_label[I,4:5]).to('cuda') # neighbor_vxc.shape = (batch,k,1)
      # train weight model
      rho = rho.to('cuda') # rho.shape = (batch, 739)
      exc = exc.to('cuda')
      vxc = vxc.to('cuda')
      rho.requires_grad = True
      x = torch.cat([torch.unsqueeze(rho, dim = 1), neighbor], dim = 1) # x.shape = (batch, k+1, 739)
      weights = model(x) # weights.shape = (batch, k, 1)
      pred_exc = torch.sum(weights * neighbor_exc, dim = (1,2)) # pred_exc.shape = (batch)
      pred_vxc = autograd.grad(torch.sum(rho[:,rho.shape[1]//2] * pred_exc), rho, create_graph = True)[0][:,rho.shape[1]//2] + pred_exc # pred_vxc.shape = (batch)
      loss1 = mae(exc, pred_exc)
      loss2 = mae(vxc, pred_vxc)
      loss = loss1 + loss2
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(loader) + step
      if global_steps % 1000 == 0:
        print(f'Step #{global_steps} exc MAE:{loss1} vxc MAE:{loss2} lr: {scheduler.get_last_lr()[0]}')
        tb_writer.add_scalar('exc loss', loss1, global_steps)
        tb_writer.add_scalar('vxc loss', loss2, global_steps)
    ckpt = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler}
    save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()

if __name__ == "__main__":
  add_options()
  app.run(main)
