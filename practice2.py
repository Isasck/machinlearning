import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd
from torchmetrics import Accuracy

X, y = make_moons(
    n_samples=1000,
    noise=0.03,
    random_state=42
)

print(X[:5], y[:5])
print(X.shape, y.shape)

moons = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "label": y
})

print(moons.head(10))
print(moons.label.value_counts())

X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
print(f"Shape for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

device = "cuda" if torch.cuda.is_available() else "cpu"


class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
        # self.layer_1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        # self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)  # extra layer
        # self.layer_3 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        # self.layer_4 = nn.Linear(in_features=hidden_units, out_features=output_features)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        return self.linear_layer_stack(x)
        # return self.layer_4(self.tanh(self.layer_3(self.tanh(self.layer_2(self.tanh((self.layer_1(x))))))))

model = MoonModel(input_features=2, output_features=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03)

acc_fn = Accuracy(task="binary").to(device)
epochs = 2000

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model.train().to(device)
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = acc_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} Accuracy: {acc} | Test loss: {test_loss} Test accuracy: {test_accuracy}")

    if test_accuracy > 0.96:
        print(f"Stopping training at epoch {epoch} because test accuracy ({test_accuracy:.2f}) > 0.96")
        break