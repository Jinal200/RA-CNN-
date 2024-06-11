# config.py

import os

# Data directory
data_dir = 'D:/RA/dataForCNN-20240516T234355Z-001/dataforCNN'

# Model hyperparameters
batch_size = 3
num_epochs = 20
learning_rate = 0.001

# Model architecture hyperparameters
conv1_out_channels = 16
conv1_kernel_size = 5
conv1_padding = 2

resblock_kernel_size = 5
resblock_padding = 2

upblock_kernel_size = 5
upblock_padding = 2

conv2_out_channels = 1
conv2_kernel_size = 5
conv2_padding = 2

BASE_LR = 0.001
MAX_LR = 0.01
STEP_SIZE_UP = 10
MODE = 'triangular2'
