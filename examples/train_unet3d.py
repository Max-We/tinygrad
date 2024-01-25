#!/usr/bin/env python3
import glob
import os

import numpy as np
from PIL import Image

from examples.mlperf.losses import dice_ce_loss
from extra.datasets.kits19 import get_val_cases, transform
from extra.models.unet3d import UNet3D
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.training import train, evaluate

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
    val_cases_list = get_val_cases()
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

# from training/image_segmentation/pytorch/run_and_time.sh
MAX_EPOCHS = 4000
QUALITY_THRESHOLD = "0.908"
START_EVAL_AT = 1000
EVALUATE_EVERY = 20
LEARNING_RATE = 0.8
LR_WARMUP_EPOCHS = 200
DATASET_DIR = "/data"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1

# from mlcommons/training_policies/blob/master/training_rules.adoc#91-hyperparameters
MOMENTUM = 0.9

if __name__ == "__main__":
  # returns all file paths to train / val data
  imgs_train, imgs_val, lbls_train, lbls_val = get_data_split("/content/drive/MyDrive/AI/kits19/preprocessed", 1, 0)

  model = UNet3D()
  optimizer = optim.SGD(get_parameters(model), lr=LEARNING_RATE, momentum=MOMENTUM)

  for epoch in range(1, MAX_EPOCHS + 1):

    # - lr warmup?
    loss_value = None
    optimizer.zero_grad()

    with Tensor.train():
      for i in range(len(imgs_train)):
        # - Batching?

        # transform
        print("Transform")
        image, label = transform(np.load(imgs_train[i]), np.load(lbls_train[i]))

        # tensor
        # - requires_grad?
        # - expand during preprocess?
        image, label = np.expand_dims(image, axis=0), np.expand_dims(label, axis=0)
        image, label = Tensor(image, requires_grad=False, dtype=dtypes.float), Tensor(label, dtype=dtypes.uint8)

        print("Forward")
        out = model(image)

        print("Loss")
        loss = dice_ce_loss(out, label)
        loss.backward()
        optimizer.step()

        print("Loss", loss.realize())

    # train(model, X_train, Y_train, optimizer, 100, BS=BATCH_SIZE, transform=transform)
    # evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
    # lr /= 1.2
    # print(f'reducing lr to {lr:.7f}')
