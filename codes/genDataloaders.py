import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# Hyperparameters for data preparation
BATCH_SIZE = 3
VAL_SPLIT = 0.1
TEST_SPLIT = 0.15
data_folder = 'D:/RA/dataForCNN-20240516T234355Z-001/dataforCNN'

# Dataset class
class NPYDataset(Dataset):
    def __init__(self, input_files, output_files):
        self.input_files = input_files
        self.output_files = output_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_data = np.load(self.input_files[idx])
        output_data = np.load(self.output_files[idx])
        return input_data, output_data

def prepare_datasets(data_dir, val_split, test_split):
    input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_SAD.npy')]
    output_files = [os.path.join(data_dir, f.replace('_SAD.npy', '_rho.npy')) for f in input_files]
    
    if not input_files or not output_files:
        raise ValueError(f"No matching .NPY files found in directory: {data_dir}")

    # Ensure files are paired correctly
    input_files.sort()
    output_files.sort()

    # Shuffle files
    combined = list(zip(input_files, output_files))
    random.shuffle(combined)
    input_files[:], output_files[:] = zip(*combined)

    # Split files into train, validation, and test
    total_files = len(input_files)
    test_size = int(test_split * total_files)
    val_size = int(val_split * (total_files - test_size))
    train_size = total_files - test_size - val_size

    # Split lists
    train_input_files = input_files[:train_size]
    train_output_files = output_files[:train_size]
    val_input_files = input_files[train_size:train_size + val_size]
    val_output_files = output_files[train_size:train_size + val_size]
    test_input_files = input_files[train_size + val_size:]
    test_output_files = output_files[train_size + val_size:]

    train_dataset = NPYDataset(train_input_files, train_output_files)
    val_dataset = NPYDataset(val_input_files, val_output_files)
    test_dataset = NPYDataset(test_input_files, test_output_files)

    return train_dataset, val_dataset, test_dataset

def prepare_dataloaders():
    train_dataset, val_dataset, test_dataset = prepare_datasets(DATA_DIR, VAL_SPLIT, TEST_SPLIT)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
