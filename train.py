import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

train_set = datasets.CIFAR10(
    root="data/", train=True, transform=my_transform, download=True
)

test_set = datasets.CIFAR10(
    root="data/", train=False, transform=my_transform, download=True
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

test_loader = DataLoader(train_set, batch_size=128)

in_channels = 3
num_classes = 10
num_epochs = 5

model = CNN(in_channels, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))


for epoch in range(num_epochs):
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        scores = model(imgs)

        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"loss: {loss}, epoch: {epoch + 1}")

model.eval()

n_correct = 0
n_samples = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device=device)
        labels = labels.to(device=device)

        scores = model(imgs)

        predictions = scores.argmax(dim=1)

        n_samples += predictions.size(0)
        n_correct += (predictions == labels).sum()

    model.train()

print(f"Accuracy: {(n_correct / n_samples) * 100:.2f}")
