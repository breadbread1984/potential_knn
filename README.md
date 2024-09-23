# Introduction

this project predict potential with KNN

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Download dataset

```shell
mkdir dataset
bash download_datasets.sh
```

## Training model

```shell
python3 train.py --trainset <path/to/trainset/dir> --evalset <path/to/evalset/dir>
```
