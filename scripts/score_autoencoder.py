"""Compute the autoencoder reconstruction error for every image.

Writes data/ae_scores.npy — one number per image, in h5 order, so it indexes
with the same split as everything else.

Run:  python scripts/score_autoencoder.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from src.autoencoder import Autoencoder
from src.common import DATA_DIR, H5_PATH, MODELS_DIR

BOTTLENECK = 8
BATCH = 32

CKPT_PATH = os.path.join(MODELS_DIR, 'autoencoder_b%d.pth' % BOTTLENECK)
OUT_PATH = os.path.join(DATA_DIR, 'ae_scores.npy')

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

model = Autoencoder(bottleneck=BOTTLENECK)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model = model.to(device)
model.eval()

with h5py.File(H5_PATH, 'r') as f:
    n_images = f['images'].shape[0]
    errors = np.zeros(n_images)

    with torch.no_grad():
        for i in range(0, n_images, BATCH):
            batch = f['images'][i:i + BATCH].astype(np.float32) / 255.0
            # transpose only changes strides, so copy to a contiguous layout:
            # the view() in Autoencoder.forward needs it. During training the
            # DataLoader's collate does this for free by stacking the batch.
            batch = np.ascontiguousarray(np.transpose(batch, (0, 3, 1, 2)))
            images = torch.from_numpy(batch).to(device)

            outputs = model(images)
            mse = ((images - outputs) ** 2).mean(dim=(1, 2, 3))
            errors[i:i + BATCH] = mse.cpu().numpy()

            if i % 3200 == 0:
                print('  %d/%d' % (i, n_images))

np.save(OUT_PATH, errors)
print('Saved: %s  (mean error %.6f)' % (OUT_PATH, errors.mean()))
