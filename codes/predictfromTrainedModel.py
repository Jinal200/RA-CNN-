import os
import numpy as np
import torch

# the directory where your test data is located
testImageLocation = 'D:/RA/dataForCNN-20240516T234355Z-001/dataforCNN/'  # Change this path accordingly
predictionFolder = 'D:/RA/predicted_data"  # Change this path accordingly

# list of all test files ending with '_SAD.npy'
files = os.listdir(testImageLocation)
files = [file for file in files if file.endswith('_SAD.npy')]

# Load the trained model
model = ResNet3D()
model.load_state_dict(torch.load("C:\\bestmodel"))

# Set the model to evaluation mode
model.eval()

# Iterate over each test file
for file in files:
    SAD_file = file
    string = file.replace('_SAD.npy', '')
    rho_file = file.replace('_SAD.npy', '_rho.npy')
    
    # Load the SAD data
    SAD = np.load(os.path.join(testImageLocation, SAD_file))
    
    # Convert SAD data to torch tensor
    SAD_tensor = torch.from_numpy(SAD)
    SAD_tensor = SAD_tensor.unsqueeze(0)  # Add batch dimension
    SAD_tensor = SAD_tensor.unsqueeze(0)  # Add channel dimension
    
    # Perform inference
    predRho = model(SAD_tensor)
    predRho = predRho.squeeze().detach().numpy()
    
    # Load actual rho data
    actRho = np.load(os.path.join(testImageLocation, rho_file))
    
    # Save predicted rho
    np.save(os.path.join(predictionFolder, f"{string}_rho_predicted.npy"), predRho)
    
    print(f"Processed: {file}")

print("Inference and prediction saving completed.")
