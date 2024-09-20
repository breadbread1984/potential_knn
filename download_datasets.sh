#!/bin/bash

python3 create_dataset.py --smiles CC --bond_dist 0.6 &
python3 create_dataset.py --smiles CC --bond_dist 0.7 &
python3 create_dataset.py --smiles CC --bond_dist 0.8 &
python3 create_dataset.py --smiles CC --bond_dist 0.9 &
python3 create_dataset.py --smiles CC --bond_dist 1.0 &
python3 create_dataset.py --smiles CC --bond_dist 1.1 &
python3 create_dataset.py --smiles CC --bond_dist 1.2 &
python3 create_dataset.py --smiles CC --bond_dist 1.3 &
python3 create_dataset.py --smiles CC --bond_dist 1.4 &
python3 create_dataset.py --smiles CC --bond_dist 1.5 &
python3 create_dataset.py --smiles C --bond_dist 0.6 &
python3 create_dataset.py --smiles C --bond_dist 0.7 &
python3 create_dataset.py --smiles C --bond_dist 0.8 &
python3 create_dataset.py --smiles C --bond_dist 0.9 &
python3 create_dataset.py --smiles C --bond_dist 1.0 &
python3 create_dataset.py --smiles C --bond_dist 1.1 &
python3 create_dataset.py --smiles C --bond_dist 1.2 &
python3 create_dataset.py --smiles C --bond_dist 1.3 &
python3 create_dataset.py --smiles C --bond_dist 1.4 &
python3 create_dataset.py --smiles C --bond_dist 1.5 &
wait
