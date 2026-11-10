"""Shared settings: paths, the anomaly definition, and THE data split.

Every script imports the split from here, so there is exactly one definition of
it in the project. That matters more than it sounds: the classifier doubles as
the feature extractor for the anomaly detectors, so if the detectors evaluated on
a different split, their evaluation images would be ones whose feature vectors
had already been optimised with knowledge of their labels.
"""
import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')
FIGURES_DIR = os.path.join(ROOT, 'figures')
DOCS_DIR = os.path.join(ROOT, 'docs')

H5_PATH = os.path.join(DATA_DIR, 'Galaxy10 DECaLS.h5')

CLASS_NAMES = [
    'Disturbed', 'Merging', 'Round Smooth', 'In-between Smooth', 'Cigar Shaped',
    'Barred Spiral', 'Unbarred Tight Spiral', 'Unbarred Loose Spiral',
    'Edge-on without Bulge', 'Edge-on with Bulge',
]

# Classes 0 (Disturbed) and 1 (Merging) count as anomalous, 2-9 as normal.
# This is the central decision of the whole project, so it lives here in code
# rather than hidden inside a pre-computed .npy file.
ANOMALOUS_CLASSES = [0, 1]
NORMAL_CLASSES = [2, 3, 4, 5, 6, 7, 8, 9]

# Seeds for every method that takes a random subsample.
SEEDS = [0, 1, 2, 3, 4]


def load_labels():
    """Class label for every image, in h5 order."""
    with h5py.File(H5_PATH, 'r') as f:
        return np.array(f['ans'])


def is_anomalous(labels):
    return np.isin(labels, ANOMALOUS_CLASSES)


def master_split(labels):
    """The one split: 80/20, stratified, random_state=42.

    Exactly the split the classifier is trained on in src/train_classifier.py.
    So "test" means "the classifier never saw this" — for the anomaly detectors
    too, since they use that classifier as their feature extractor.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )
    return np.sort(train_idx), np.sort(test_idx)


def reference_pool(labels, train_idx):
    """Normal galaxies from the train split — the only data a detector may fit on."""
    return train_idx[~is_anomalous(labels[train_idx])]


def load_features(backbone):
    """Loads data/feat_<backbone>.npz: feats (N,512), labels (N,), logits (N,10)."""
    path = os.path.join(DATA_DIR, 'feat_%s.npz' % backbone)
    if not os.path.exists(path):
        raise FileNotFoundError(path + ' not found. Run evaluation/extract_features.py first')
    return np.load(path)
