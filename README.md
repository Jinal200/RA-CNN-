# RA-CNN-
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        return input_data, output_data  # Return both input and output data

# Train data and split into train and test sets
data_dir = 'D:/RA/dataForCNN-20240516T234355Z-001/dataforCNN'

# Gather input and output files
input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_SAD.npy')]
output_files = [os.path.join(data_dir, f.replace('_SAD.npy', '_rho.npy')) for f in input_files]

if not input_files or not output_files:
    raise ValueError(f"No matching .NPY files found in directory: {data_dir}")

# Ensure input and output files are properly matched
assert all([os.path.basename(f).replace('_SAD.npy', '_rho.npy') in os.listdir(data_dir) for f in input_files]), "Mismatch in input and output files"

# Split input_files and output_files into 85% train and 15% test
train_size = int(0.85 * len(input_files))
test_size = len(input_files) - train_size
train_input_files, test_input_files = random_split(input_files, [train_size, test_size])
train_output_files, test_output_files = random_split(output_files, [train_size, test_size])

# Datasets and DataLoaders
train_dataset = NPYDataset(train_input_files, train_output_files)
test_dataset = NPYDataset(test_input_files, test_output_files)

# Dynamically set batch size
total_images = len(train_input_files)
batch_size = min(10, total_images // 2)  # Ensure the batch size is not too large

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define a residual block with Conv3d, InstanceNorm3d, and PReLU using circular padding
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, padding_mode='circular')
        self.inorm1 = nn.InstanceNorm3d(in_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, padding_mode='circular')
        self.inorm2 = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.inorm1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.inorm2(out)
        out += residual  # Add the residual connection
        return out

# Define an upscaling block with Conv3d and PReLU using circular padding
class UpscalingBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpscalingBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2, padding_mode='circular')
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.prelu(self.conv(x))
        return out

# 3D CNN model
class Advanced3DCNN(nn.Module):
    def __init__(self):
        super(Advanced3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=5, padding=2, padding_mode='circular')
        self.prelu1 = nn.PReLU()
        self.resblock1 = ResidualBlock(16)
        self.upblock1 = UpscalingBlock(16)
        self.upblock2 = UpscalingBlock(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=5, padding=2, padding_mode='circular')
        self.relu = nn.ReLU()  # Final ReLU activation

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.resblock1(x)
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.relu(self.conv2(x))  # Apply ReLU at the end
        return x

# Function to calculate loss on the dataset
def calculate_loss(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.unsqueeze(1).float().to(device)  # Shape: [batch_size, 1, 66, 66, 66]
            targets = targets.unsqueeze(1).float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(inputs)
    return total_loss / len(data_loader.dataset)

# Instantiate the model, define loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Advanced3DCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the CyclicLR scheduler
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=5, mode='triangular2')

# Train the model
num_epochs = 20  # Increase number of epochs for better convergence
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1).float().to(device)  # Shape: [batch_size, 1, 66, 66, 66]
        targets = targets.unsqueeze(1).float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step()

        running_loss += loss.item()
    
    average_train_loss = running_loss / len(train_loader)
    train_losses.append(average_train_loss)
    
    # Calculate validation loss
    val_loss = calculate_loss(model, criterion, test_loader, device)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.6f}, Validation Loss: {val_loss:.6f}')

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_loss = val_loss

    # Early stopping condition (uncomment to use)
    # if epoch > 1 and abs(val_losses[-1] - val_losses[-2]) < 1e-5:
    #     print("Early stopping as loss convergence is minimal")
    #     break

# Plotting the training and validation loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Print biases of the model
print("Model Biases:")
for name, param in model.named_parameters():
    if 'bias' in name:
        print(f"{name}: {param.data}")
