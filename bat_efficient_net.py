import datetime

import pandas as pd
import torch
import os

from torch import nn, optim
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet as EfficientNet_Baseline
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms, datasets
import numpy as np
from timeit import default_timer as timer

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
TRAIN = "train"
TEST = "test"
VALIDATION = "validation"
num_of_classes = 3

ADAM = "Adam"
SGD = "SGD"
PATH = r"C:\olesya\new_bats2"
model_name = "efficientnet-b0"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()])

train_set = datasets.ImageFolder(os.path.join(PATH, TRAIN), transform=transform)

validation_set = datasets.ImageFolder(os.path.join(PATH, VALIDATION), transform=test_transform)

test_set = datasets.ImageFolder(os.path.join(PATH, TEST), transform=test_transform)


weights = EfficientNet_B0_Weights.DEFAULT
model = EfficientNet_Baseline.from_pretrained(model_name, num_classes=num_of_classes).to(device)
# model = EfficientNet.from_pretrained(weights=weights).to(device)


# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# # Unfreeze few blocks
# for name, module in model.named_modules():
#     if name in ['_blocks.20', '_blocks.21', '_blocks.22', '_fc']:
#         for param in module.parameters():
#             param.requires_grad = True

# Unfreeze the last layer parameters
for param in model._fc.parameters():
    param.requires_grad = True

# model.fc = nn.Sequential(
#     nn.Linear(2048, 128),
#     nn.ReLU(inplace=True),
#     nn.Linear(128, 2)).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model._fc.parameters())

# # Step 2: Initialize the inference transforms
preprocess = weights.transforms()


def train(batch_size=32, epochs=10):
    losses = []
    accurancies = []
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs_on_device = inputs.to(device)
            labels_on_device = labels.to(device)
            # optimizer.zero_grad()
            outputs = model(inputs_on_device)
            loss = criterion(outputs, labels_on_device)
            loss.backward()
            # optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_on_device.data)

        epoch_loss = running_loss / len(train_set)
        losses.append(epoch_loss)
        epoch_acc = running_corrects.double().to(device="cpu") / len(train_set)
        accurancies.append(epoch_acc)

        print('{} {} loss: {:.4f}, acc: {:.4f} %'.format(TRAIN,
                                                         epoch,
                                                         epoch_loss,
                                                         epoch_acc * 100))
        validation_acc = test(batch_size=batch_size, validate=True)
        print('{} {} val acc: {:.4f} %'.format(TRAIN,
                                               epoch,
                                               validation_acc * 100))

    return losses, accurancies


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def my_plot(epochs, loss):
    plt.plot(epochs, loss)


def test(batch_size=32, show=False, validate=False):
    running_loss = 0.0
    running_corrects = 0
    if validate:
        test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    accurancies = []
    losses = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_set)
    print(f"Running_corrects: {running_corrects}")
    epoch_acc = running_corrects.double() / len(test_set)

    print('Epoch {} loss: {:.4f}, acc: {:.4f} %'.format("test",
                                                        epoch_loss,
                                                        epoch_acc * 100))

    return epoch_acc


def main():
    title = ["model", "batch_size", "learning_rate", "optimizer", "augmentation", "epochs", "accuracy"]
    data = []
    batch_size = 32

    combinations = [
        # (1, 0.001, ADAM, True),
        (10, 0.001, ADAM, True),
        # (40, 0.001, ADAM, True),
        # (80, 0.001, ADAM, True),
        # (80, 0.001, SGD, True),
        # (100, 0.001, SGD, True),
        # (120, 0.0001, SGD, True)
    ]
    for i in combinations:
        epoch = i[0]
        lr_rate = i[1]
        optimizer = i[2]
        augmentation = i[3]
        print(f"Epoch: {epoch}, LR: {lr_rate}, Optimizer: {optimizer}, Augmentation: {augmentation}")
        start_time = timer()
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M")
        print(dt_string)
        fig_name = f"{model_name}_loss_{dt_string}.png"
        print(fig_name)
        losses, accuracies = train(batch_size, epoch)
        end_time = timer()
        print("Finished training")
        print(f"Total training time: {end_time - start_time:.3f} seconds")
        print(losses)
        plt.clf()
        my_plot(np.linspace(1, epoch, epoch).astype(int), losses)
        plt.savefig(fig_name)
        plt.clf()
        fig_name = f"{model_name}_accuracy_{dt_string}.png"
        my_plot(np.linspace(1, epoch, epoch).astype(int), accuracies)
        plt.savefig(fig_name)
        accuracy = test(32)
        end_time = timer()
        print(f"Total time: {end_time - start_time:.3f} seconds")
        data.append([model, batch_size, lr_rate, optimizer, augmentation, epoch, accuracy])

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    df = pd.DataFrame(data, columns=title, dtype=str)
    df.to_csv(f"{model_name}_bats_{dt_string}.csv", index=False)


if __name__ == '__main__':
    main()
