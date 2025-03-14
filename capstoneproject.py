import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchinfo import summary
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from torchmetrics import Accuracy

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet18_Weights.DEFAULT
    # data_transform = weights.transforms()
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])
    # print(weights)

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001

    train_data = datasets.CIFAR10(root="./datasets",
                                  transform=manual_transforms,
                                  target_transform=None,
                                  download=True,
                                  train=True)
    test_data = datasets.CIFAR10(root="./datasets",
                                 transform=manual_transforms,
                                 target_transform=None,
                                 download=True,
                                 train=False)
    train_dataloader = DataLoader(dataset=train_data,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  num_workers=os.cpu_count())
    test_dataloader = DataLoader(dataset=test_data,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 num_workers=os.cpu_count())

    class_names = train_data.classes
    output_shape = len(class_names)

    model = resnet18(weights=weights).to(device)

    # print(class_names, len(class_names))

    for params in model.parameters():
        params.requires_grad = False

    model.fc = nn.Linear(in_features=512,
                         out_features=10,
                         bias=True)

    # print(summary(model=model,
    #               input_size=(32, 3, 224, 224),
    #               # col_names=["input_size"],
    #               col_names=["input_size", "output_size", "num_params", "trainable"],
    #               col_width=20,
    #               row_settings=["var_names"]
    # ))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    accuracy_fn = Accuracy(task="multiclass", num_classes=10).to(device)

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_loss, train_acc = 0, 0
        for X, y in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', unit='batch'):
            model.train()
            X, y = X.to(device), y.to(device)
            y_pred = model(X).to(device)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_pred.argmax(dim=1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        test_loss, test_acc = 0, 0

        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(test_pred.argmax(dim=1), y)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print("---")

if __name__ == '__main__':
    main()