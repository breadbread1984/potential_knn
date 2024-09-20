#!/usr/bin/python3

from absl import flags, app
from os import walk
from os.path import join, exists, splitext
import numpy as np

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'path to trainset')

def convert(dir_path):
  file_list = list()
  rho = list()
  for root, dirs, files in walk(dir_path):
    for f in files:
      stem, ext = splitext(f)
      if ext != '.npy': break
      file_list.append(join(root, f))
      data = np.load(join(root, f))
      assert data.shape[1] == 769 + 3 + 1 + 1
      rho.append(data[:,:739])
  rho = np.concatenate(rho, axis = 0)
  np.save(join(dir_path,'trainset_rho.npy'), rho)
  del rho
  label = list()
  for file_path in file_list:
    data = np.load(file_path)
    label.append(data[:,769:])
  label = np.concatenate(label, axis = 0)
  np.save(join(dir_path,'trainset_label.npy'), label)

def main(unused_argv):
  convert(FLAGS.trainset)

if __name__ == "__main__":
  add_options()
  app.run(main)

