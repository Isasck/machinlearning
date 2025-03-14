import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # Define fc1
        self.fc2 = nn.Linear(in_features=128, out_features=10)   # Define fc2

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Use fc1
        x = self.fc2(x)          # Use fc2
        return x

# Create an instance of the model
model = NN()

# Example input (batch of 64 images, each with 784 features)
data = torch.randn(64, 784)

# Forward pass
scores = model(data)
print(scores.shape)  # Should output: torch.Size([64, 10])