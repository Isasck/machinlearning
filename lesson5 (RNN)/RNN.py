import torch
import torch.nn as nn   # Provides neural network layers and loss functions
import torch.optim as optim # Contains optimization algorithms like Adam
import torch.nn.functional as F  # Contains activation functions like ReLU
from torch.utils.data import DataLoader  # Helps in loading and batching datasets
import torchvision.datasets as datasets  # Provides access to standard datasets like MNIST
import torchvision.transforms as transforms # Provides image transformations

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

"""TODO: Create a simple RNN"""
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Propagation
        # out, _ = self.rnn(x, h0)      # For RNN and GRU
        out, _ = self.lstm(x, (h0, c0))      # For LSTM
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)
        out = self.fc(out[:,-1,:])
        return out


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Load Data
train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length=sequence_length).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights


# Check accuracy on training & test to see how good our model is

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # 64x10, find maximum value of the ten digits
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)