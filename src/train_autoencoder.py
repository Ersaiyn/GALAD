"""Train the autoencoder on NORMAL galaxies from the train split only.

Training on normal galaxies alone is the whole idea: the model learns to
reconstruct ordinary morphologies, and anomalies should then reconstruct badly.
Restricting it to the train split keeps the test set genuinely held out, the
same way it is held out for the classifier.

Run:  python src/train_autoencoder.py
Change BOTTLENECK below to try a different latent size.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.autoencoder import Autoencoder
from src.common import H5_PATH, MODELS_DIR, master_split, reference_pool
from src.dataset import GalaxyDataset

BOTTLENECK = 8
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OUT_PATH = os.path.join(MODELS_DIR, 'autoencoder_b%d.pth' % BOTTLENECK)

with h5py.File(H5_PATH, 'r') as f:
    labels = np.array(f['ans'])
    train_idx, _ = master_split(labels)
    pool = reference_pool(labels, train_idx)
    images = np.array(f['images'][pool])

print('training on %d normal galaxies from the train split (bottleneck=%d)' %
      (len(pool), BOTTLENECK))

loader = DataLoader(GalaxyDataset(images, labels[pool], augment=True),
                    batch_size=BATCH_SIZE, shuffle=True)

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = Autoencoder(bottleneck=BOTTLENECK).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images_batch, _ in loader:
        images_batch = images_batch.float().to(device)

        outputs = model(images_batch)
        loss = criterion(outputs, images_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch [%d/%d] Loss: %.6f' % (epoch + 1, EPOCHS, running_loss / len(loader)))

torch.save(model.state_dict(), OUT_PATH)
print('Saved: %s' % OUT_PATH)
