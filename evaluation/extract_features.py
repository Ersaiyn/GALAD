"""Compute 512-d features (and logits) for all 17736 images, for each backbone.

Writes data/feat_<name>.npz with keys feats, labels, logits.
The order is the h5 order, so the arrays index with the same split from
src/common.py.

Backbones:
  classifier        models/classifier.pth — augmented, 81.5% train / 78.3% test.
                    This is the model the methodology describes. Use this one.
  classifier_noaug  models/classifier_noaug.pth — un-augmented, 94.8% / 73.5%.
                    The ablation that shows what augmentation buys.
  imagenet          ResNet-18 with ImageNet weights, no fine-tuning. The control:
                    shows how much signal the fine-tuning on labels provides.
  imagenet_norm     the same, but with proper ImageNet normalisation — so the
                    control cannot be blamed on this project having no mean/std.

The fine-tuned backbones get img/255 with no mean/std — exactly how they were
trained.

Run:  python evaluation/extract_features.py
About 17 minutes on an M-series Mac. Files already computed are skipped.
"""
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.common import DATA_DIR, H5_PATH, MODELS_DIR

BATCH = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# name -> (checkpoint path or None, whether ImageNet normalisation is needed)
BACKBONES = [
    ('classifier', os.path.join(MODELS_DIR, 'classifier.pth'), False),
    ('classifier_noaug', os.path.join(MODELS_DIR, 'classifier_noaug.pth'), False),
    ('imagenet', None, False),
    ('imagenet_norm', None, True),
]

device = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)

with h5py.File(H5_PATH, 'r') as f:
    n_images = f['images'].shape[0]
    labels = np.array(f['ans'])

    for name, ckpt, normalize in BACKBONES:
        out_path = os.path.join(DATA_DIR, 'feat_%s.npz' % name)
        if os.path.exists(out_path):
            print('%s: already computed, skipping' % name)
            continue
        if ckpt is not None and not os.path.exists(ckpt):
            print('%s: no checkpoint at %s, skipping' % (name, ckpt))
            continue

        model = models.resnet18(weights='DEFAULT')
        last_layer = None
        if ckpt is not None:
            model.fc = nn.Linear(512, 10)
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            last_layer = model.fc          # remember the last layer
        model.fc = nn.Identity()           # now the model outputs 512 features
        model = model.to(device)
        model.eval()

        feats = np.zeros((n_images, 512), dtype=np.float32)
        logits = np.zeros((n_images, 10), dtype=np.float32)

        start = time.time()
        with torch.no_grad():
            for i in range(0, n_images, BATCH):
                batch = f['images'][i:i + BATCH].astype(np.float32) / 255.0
                batch = np.transpose(batch, (0, 3, 1, 2))
                images = torch.from_numpy(batch).to(device)
                if normalize:
                    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
                    images = (images - mean) / std

                features = model(images)
                feats[i:i + BATCH] = features.cpu().numpy()
                if last_layer is not None:
                    # logits are just the last layer applied to the features
                    logits[i:i + BATCH] = last_layer(features.cpu()).numpy()

                if i % 3200 == 0:
                    print('  %s: %d/%d  %.0f s' % (name, i, n_images, time.time() - start))

        if last_layer is not None:
            np.savez(out_path, feats=feats, labels=labels, logits=logits)
        else:
            np.savez(out_path, feats=feats, labels=labels)
        print('%s: done in %.0f s -> %s' % (name, time.time() - start, out_path))

print('All backbones computed.')
