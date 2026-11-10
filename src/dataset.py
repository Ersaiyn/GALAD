"""Dataset wrapper for Galaxy10 DECaLS.

Note on preprocessing: images are scaled to [0,1] and nothing else. ImageNet
mean/std normalisation is NOT applied, even though the backbone is ImageNet
pre-trained. The checkpoints were trained this way, so changing it would
invalidate them. See REPORT.md section 7.
"""
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class GalaxyDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ]) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index] / 255.0
        image = torch.FloatTensor(np.transpose(image, (2, 0, 1)))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
