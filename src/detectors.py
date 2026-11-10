"""The four detectors and the three reference sets they fit on.

One convention everywhere: the higher the score, the more anomalous. That way
roc_auc_score can be called the same way for every method.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

NORMAL_CLASSES = [2, 3, 4, 5, 6, 7, 8, 9]


# ---------------------------------------------------------------- reference sets
# A detector may only be fitted on normal galaxies from the train split.
# Which normal galaxies turned out to matter more than the choice of algorithm.

def ref_full(feats, labels, pool, seed=None):
    """Every normal galaxy in the train split (11841 of them)."""
    return feats[pool]


def ref_balanced(feats, labels, pool, seed=0):
    """Equal count from each normal class, sampled at random.

    The size is capped by the rarest normal class: Cigar Shaped has only 267
    galaxies in the train split, so 8 * 267 = 2136. Asking for more per class
    would silently return fewer than requested.
    """
    rng = np.random.default_rng(seed)
    chosen = []
    per_class = [pool[labels[pool] == c] for c in NORMAL_CLASSES]
    k = min(len(idx) for idx in per_class)
    for idx in per_class:
        chosen.append(rng.choice(idx, k, replace=False))
    return feats[np.sort(np.concatenate(chosen))]


def ref_imbalanced(feats, labels, pool, seed=None, n=3000):
    """Reference set taken in dataset order, kept as an ablation.

    Images in the h5 file are sorted by class, so "the first 3000 normal ones"
    means only Round Smooth (2116) and In-between Smooth (884). Six of the
    eight normal classes are not there at all. A detector fitted on this gives
    Edge-on galaxies a higher score than actual mergers — see REPORT.md 4.3.
    """
    class_sorted = pool[np.argsort(labels[pool], kind='stable')]
    return feats[class_sorted[:n]]


# -------------------------------------------------------------------- detectors

def knn_score(reference, query, k=5):
    """Mean distance to the k nearest normal neighbours in feature space."""
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(reference)
    distances, _ = knn.kneighbors(query)
    return distances.mean(axis=1)


def fit_pca(reference):
    """PCA on the reference set. Fit once with all components: taking the first
    n columns of the transform is the same as fitting PCA(n)."""
    pca = PCA(n_components=min(reference.shape), random_state=42)
    pca.fit(reference)
    return pca


def isolation_forest_score(reference, query, seed=0, n_components=None, pca=None):
    """Isolation Forest, optionally in the space after PCA.

    Without PCA it works poorly (AUROC 0.69). The reason is how sklearn builds
    the trees: ExtraTreeRegressor(max_features=1), so every split cuts on ONE
    random feature. On raw correlated features such a cut says almost nothing.
    PCA rotates the axes so they become uncorrelated, and then the cuts start
    to work. It is about the rotation of the axes, not about how much variance
    is kept: PCA(32) keeps 99.4% of the variance and is still worse than PCA(450).
    """
    if n_components is not None:
        reference = pca.transform(reference)[:, :n_components]
        query = pca.transform(query)[:, :n_components]
    forest = IsolationForest(n_estimators=200, contamination=0.1,
                             random_state=seed, n_jobs=-1)
    forest.fit(reference)
    return -forest.score_samples(query)


def confidence_score(logits):
    """1 - max softmax. Does not work here: the classifier has its own output
    neurons for both "anomalous" classes, so they are not out-of-distribution
    for it at all."""
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = shifted / shifted.sum(axis=1, keepdims=True)
    return 1 - softmax.max(axis=1)
