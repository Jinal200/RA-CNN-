# main.py

import config
from gendataloaders import prepare_dataloaders
from config import BASE_LR, MAX_LR, STEP_SIZE_UP, MODE
from cnn_model import ResNet3D, calculate_loss
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Datasets and DataLoaders
train_loader, test_loader = prepare_dataloaders(config.data_dir, config.batch_size)

# Instantiate the model, define loss function and optimizer
model = Advanced3DCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=BASE_LR,
    max_lr=MAX_LR,
    step_size_up=STEP_SIZE_UP,
    mode=MODE
    
# Train the model
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(config.num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1).float()  # Shape: [batch_size, 1, 66, 66, 66]
        targets = targets.unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    average_train_loss = running_loss / len(train_loader)
    train_losses.append(average_train_loss)
    
    # Calculate validation loss
    val_loss = calculate_loss(model, criterion, test_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{config.num_epochs}, Train Loss: {average_train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    # Save the model if the validation loss is lower than the previous validation loss
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'C:\\bestmodel')
        best_val_loss = val_loss

