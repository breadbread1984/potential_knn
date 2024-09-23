# Introduction

this project predict potential with KNN

# Usage

## Install faiss

```shell
git clone git@github.com:facebookresearch/faiss.git
cd faiss
mkdir build && cd build
cmake .. -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.3 -DBUILD_TESTING=OFF
make -j32
cd faiss/python
python3 setup.py bdist
cd dist
python3 -m pip install faiss-1.8.0-py3-none-any.whl
```

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Download dataset

```shell
mkdir dataset
bash download_datasets.sh
```

## Train model

```shell
python3 train.py --trainset <path/to/trainset/dir> --evalset <path/to/evalset/dir>
```

## Evaluate model

```shell
python3 evaluate.py --trainset <path/to/trainset/dir> --evalset <path/to/evalset/dir>
```
