#!/usr/bin/env python3
import glob
import os

import numpy as np
from PIL import Image

from extra.models.unet3d import UNet3D
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.training import train, evaluate

MAX_EPOCHS = 4000
QUALITY_THRESHOLD = "0.908"
START_EVAL_AT = 1000
EVALUATE_EVERY = 20
LEARNING_RATE = "0.8"
LR_WARMUP_EPOCHS = 200
DATASET_DIR = "/data"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1

# MAX_EPOCHS = 500
# STEPS = 250
# QUALITY_THRESHOLD = "0.908"
# EVALUATE_EVERY = 20
# LEARNING_RATE = 1e-4
# DATASET_DIR = "/data"
# BATCH_SIZE = 2

def list_files_with_pattern(path, files_pattern):
  data = sorted(glob.glob(os.path.join(path, files_pattern)))
  assert len(data) > 0, f"Found no data at {path}"
  return data

def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]


def get_data_split(path: str, num_shards: int, shard_id: int):
    with open("evaluation_cases.txt", "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    imgs_val, lbls_val = split_eval_data(imgs_val, lbls_val, num_shards, shard_id)
    return imgs_train, imgs_val, lbls_train, lbls_val


if __name__ == "__main__":
  # shard-id = 1 ok?
  # what is shard?
  imgs_train, imgs_val, lbls_train, lbls_val = get_data_split("PATH", 1, 1)
  print(imgs_train)

  TRANSFER = getenv('TRANSFER')
  model = UNet3D()

  lr = 5e-3

  for _ in range(5):
    optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
    # train(model, X_train, Y_train, optimizer, 100, BS=BATCH_SIZE, transform=transform)
    # evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
    # lr /= 1.2
    # print(f'reducing lr to {lr:.7f}')
