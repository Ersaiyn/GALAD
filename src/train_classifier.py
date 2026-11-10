"""Fine-tune ResNet-18 on the ten Galaxy10 morphology classes.

Augmentation is on and should stay on: without it the model memorises the
training set — 94.8% on train against 73.5% on test. With it, 81.5% / 78.3%.
The un-augmented checkpoint is kept as an ablation (REPORT.md section 3.2),
which is what AUGMENT = False produces.

All layers are trained, nothing is frozen — the ImageNet weights are a
starting point, not a fixed feature extractor.

Run:  python src/train_classifier.py
For the un-augmented variant set AUGMENT = False below.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from src.common import H5_PATH, MODELS_DIR, master_split
from src.dataset import GalaxyDataset

AUGMENT = True
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OUT_NAME = 'classifier.pth' if AUGMENT else 'classifier_noaug.pth'
OUT_PATH = os.path.join(MODELS_DIR, OUT_NAME)

with h5py.File(H5_PATH, 'r') as f:
    labels = np.array(f['ans'])
    train_idx, test_idx = master_split(labels)
    train_images = np.array(f['images'][train_idx])
    test_images = np.array(f['images'][test_idx])

train_labels = labels[train_idx]
test_labels = labels[test_idx]
print('train %d   test %d   augment=%s' % (len(train_idx), len(test_idx), AUGMENT))

train_dataset = GalaxyDataset(train_images, train_labels, augment=AUGMENT)
test_dataset = GalaxyDataset(test_images, test_labels, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_loader:
        images = images.float().to(device)
        targets = targets.long().to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print('Epoch [%d/%d] Loss: %.4f Accuracy: %.1f%%' %
          (epoch + 1, EPOCHS, running_loss / len(train_loader), 100 * correct / total))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, targets in test_loader:
        images = images.float().to(device)
        targets = targets.long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print('Test Accuracy: %.1f%%' % (100 * correct / total))

torch.save(model.state_dict(), OUT_PATH)
print('Saved: %s' % OUT_PATH)
