from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dataset = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

img = train_dataset[0][0]
label = train_dataset[0][1]

print(f"Image:\n {img}")
print(f"Label:\n {label}")
print(f"Image shape: {img.shape} -> [color_channels, height, width] (CHW)")
print(f"Label: {label} -> no shape, due to being integer")

class_names = train_dataset.classes

# for i in range(5):
#     img = train_dataset[i][0].squeeze()
#     label = train_dataset[i][1]
#     plt.figure(figsize=(3,3))
#     plt.imshow(img, cmap="gray")
#     plt.title(label)
#     plt.axis(False)

batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# for sample in next(iter(train_loader)):
#   print(sample.shape)

class CNN(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, out_channels: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 6 * 6,
                      out_features=out_channels)
        )

    def forward(self, x):
        x = self.block_1(x)
        # print(f"Output shape of conv block 1: {x.shape}")
        x = self.block_2(x)
        # print(f"Output shape of conv block 2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")
        return x


torch.manual_seed(42)
# model = CNN(in_channels=1, hidden_units=10, out_channels=10).to(device)
epochs = 5

model_cpu = CNN(in_channels=1, out_channels=10, hidden_units=10).to("cpu")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_cpu.parameters(), lr=0.1)

for epoch in tqdm(range(epochs)):
  train_loss = 0
  for batch, (X, y) in enumerate(train_loader):
    model_cpu.train()
    X, y = X.to("cpu"), y.to("cpu")
    y_pred = model_cpu(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss /= len(train_loader)

    test_loss_total = 0

    model_cpu.eval()
    with torch.inference_mode():
      for batch, (X_test, y_test) in enumerate(test_loader):
        X_test, y_test = X_test.to("cpu"), y_test.to("cpu")
        test_pred = model_cpu(X_test)
        test_loss = loss_fn(test_pred, y_test)
        test_loss_total += test_loss

      test_loss_total /= len(test_loader)

  print(f"Epoch: {epoch} | Loss: {train_loss:.3f} | Test loss: {test_loss_total:.3f}")
