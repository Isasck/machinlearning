import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=out_channels)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.classifier(x)
        return x

model = CNN(in_channels=3, out_channels=10, hidden_units=16)

input_tensor = torch.randn(8, 3, 32, 32)  # Batch size of 8
output = model(input_tensor)
print(output.shape)

