#!/usr/bin/python3

from shutil import rmtree
from os import mkdir, listdir
from os.path import exists, join, splitext
from absl import flags, app
import mysql.connector
import numpy as np
from bisect import bisect
from torch.utils.data import Dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('addr', default = '103.6.49.76', help = 'mysql service address')
  flags.DEFINE_string('user', default = 'root', help = 'user name')
  flags.DEFINE_string('password', default = '12,.zheng', help = 'password')
  flags.DEFINE_string('db', default = 'dft', help = 'database to use')
  flags.DEFINE_string('output', default = 'dataset', help = 'path to output directory')
  flags.DEFINE_string('smiles', default = 'CC', help = 'SMILES')
  flags.DEFINE_float('bond_dist', default = 1., help = 'bond distance')

def main(unused_argv):
  if not exists(FLAGS.output): mkdir(FLAGS.output)
  conn = mysql.connector.connect(
    host = FLAGS.addr,
    user = FLAGS.user,
    password = FLAGS.password,
    database = FLAGS.db)
  cursor = conn.cursor()
  samples = list()
  start = 0
  while True:
    sql = "select arr_cc, exc, vxc, gc from %s.grid_b3_with_HFx where smiles = '%s' and abs(bond_length - %f) < 1e-6 limit %d, 100" % (FLAGS.db, FLAGS.smiles, FLAGS.bond_dist - 1, start)
    try:
      cursor.execute(sql)
      rows = cursor.fetchall()
      if len(rows) == 0: break
      for row in rows:
        samples.append(np.concatenate([np.array(eval(row[0])).flatten(), np.array(eval(row[3])), [row[1],], [row[2],]], axis = 0))
      start += len(rows)
      print('bond: %f fetched: %d' % (FLAGS.bond_dist, len(samples)))
    except:
      break
  output = np.stack(samples, axis = 0)
  np.save(join(FLAGS.output, '%s_%f.npy' % (FLAGS.smiles, FLAGS.bond_dist)), output)

if __name__ == "__main__":
  add_options()
  app.run(main)
