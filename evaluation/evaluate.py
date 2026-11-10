"""Compute every number in REPORT.md under one protocol.

Run evaluation/extract_features.py first.

PROTOCOL
  split           src/common.py -> master_split: 80/20, stratified,
                  random_state=42 — the same split the classifier was trained on
  fitted on       normal galaxies from the TRAIN split only
  evaluated on    the whole TEST split: 2961 normal + 587 anomalous, i.e. on
                  images the classifier never saw
  positive class  anomalous
  metrics         AUROC and average precision (AP). AP is needed because the
                  test split carries the real 16.5% anomaly rate rather than an
                  artificial 50/50: a random detector scores AUROC 0.500, AP 0.165
  variance        every method that subsamples runs over 5 seeds, mean +- sd

k for kNN is fixed at 5 and was not tuned.

Results go to docs/results.json, which evaluation/figures.py reads.

Run:  python evaluation/evaluate.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from src.common import (CLASS_NAMES, DATA_DIR, DOCS_DIR, NORMAL_CLASSES, SEEDS,
                        is_anomalous, load_features, load_labels, master_split,
                        reference_pool)
from src.detectors import (confidence_score, fit_pca, isolation_forest_score,
                           knn_score, ref_balanced, ref_full, ref_imbalanced)

K = 5
PCA_GRID = [8, 32, 100, 200, 380, 450]

# kNN runs on every backbone, because the ImageNet ones are the control that shows
# where the detection signal comes from (REPORT.md section 5). Isolation Forest only
# runs on the main backbone — the report never uses IF numbers for the controls.
BACKBONES = ['classifier', 'classifier_noaug', 'imagenet', 'imagenet_norm']
MAIN_BACKBONE = 'classifier'

results = {}

# ============================================================ protocol
labels = load_labels()
train_idx, test_idx = master_split(labels)
pool = reference_pool(labels, train_idx)

y_eval = is_anomalous(labels[test_idx]).astype(int)
prevalence = y_eval.mean()

print('=' * 70)
print('PROTOCOL')
print('  train %d   test %d   reference pool %d' % (len(train_idx), len(test_idx), len(pool)))
print('  evaluation: %d normal + %d anomalous' % ((y_eval == 0).sum(), (y_eval == 1).sum()))
print('  prevalence %.4f  (the AP of a random detector)' % prevalence)
print('=' * 70)

results['protocol'] = {
    'n_train': len(train_idx),
    'n_test': len(test_idx),
    'n_reference_pool': len(pool),
    'n_eval_normal': int((y_eval == 0).sum()),
    'n_eval_anomalous': int((y_eval == 1).sum()),
    'prevalence': float(prevalence),
    'k': K,
}

# ============================================================ detectors per backbone
results['detection'] = {}

for backbone in BACKBONES:
    feats = load_features(backbone)['feats']
    query = feats[test_idx]
    res = {}
    print('\n--- backbone: %s ---' % backbone)

    # kNN on all normal galaxies from train (no subsampling, so no seed needed)
    scores = knn_score(ref_full(feats, labels, pool), query, K)
    res['knn_full'] = {
        'auroc': float(roc_auc_score(y_eval, scores)), 'auroc_sd': 0.0,
        'ap': float(average_precision_score(y_eval, scores)), 'ap_sd': 0.0,
    }

    # kNN on a balanced reference set, 5 seeds
    auroc_list = []
    ap_list = []
    for seed in SEEDS:
        reference = ref_balanced(feats, labels, pool, seed)
        scores = knn_score(reference, query, K)
        auroc_list.append(roc_auc_score(y_eval, scores))
        ap_list.append(average_precision_score(y_eval, scores))
    res['knn_balanced'] = {
        'auroc': float(np.mean(auroc_list)), 'auroc_sd': float(np.std(auroc_list)),
        'ap': float(np.mean(ap_list)), 'ap_sd': float(np.std(ap_list)),
    }

    # kNN on a reference set taken in dataset order (two classes of eight)
    scores = knn_score(ref_imbalanced(feats, labels, pool), query, K)
    res['knn_imbalanced'] = {
        'auroc': float(roc_auc_score(y_eval, scores)), 'auroc_sd': 0.0,
        'ap': float(average_precision_score(y_eval, scores)), 'ap_sd': 0.0,
    }

    for tag in ['knn_full', 'knn_balanced', 'knn_imbalanced']:
        print('  %-16s AUROC %.4f +- %.4f   AP %.4f' %
              (tag, res[tag]['auroc'], res[tag]['auroc_sd'], res[tag]['ap']))

    if backbone != MAIN_BACKBONE:
        results['detection'][backbone] = res
        continue

    # Isolation Forest on the raw 512 dimensions, 5 seeds
    reference = ref_balanced(feats, labels, pool, 0)
    auroc_list = []
    ap_list = []
    for seed in SEEDS:
        scores = isolation_forest_score(reference, query, seed)
        auroc_list.append(roc_auc_score(y_eval, scores))
        ap_list.append(average_precision_score(y_eval, scores))
    res['if_raw512'] = {
        'auroc': float(np.mean(auroc_list)), 'auroc_sd': float(np.std(auroc_list)),
        'ap': float(np.mean(ap_list)), 'ap_sd': float(np.std(ap_list)),
    }
    print('  %-16s AUROC %.4f +- %.4f   AP %.4f' %
          ('if_raw512', res['if_raw512']['auroc'], res['if_raw512']['auroc_sd'],
           res['if_raw512']['ap']))

    # Isolation Forest after PCA, over a grid of component counts
    pca = fit_pca(reference)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    res['if_pca'] = {}
    for n_components in PCA_GRID:
        auroc_list = []
        ap_list = []
        for seed in SEEDS:
            scores = isolation_forest_score(reference, query, seed, n_components, pca)
            auroc_list.append(roc_auc_score(y_eval, scores))
            ap_list.append(average_precision_score(y_eval, scores))
        res['if_pca'][n_components] = {
            'auroc': float(np.mean(auroc_list)), 'auroc_sd': float(np.std(auroc_list)),
            'ap': float(np.mean(ap_list)), 'ap_sd': float(np.std(ap_list)),
            'cumulative_variance': float(cumulative_variance[n_components - 1]),
        }
        print('  IF+PCA(%3d)      AUROC %.4f +- %.4f   AP %.4f   variance %.4f' %
              (n_components, res['if_pca'][n_components]['auroc'],
               res['if_pca'][n_components]['auroc_sd'],
               res['if_pca'][n_components]['ap'],
               res['if_pca'][n_components]['cumulative_variance']))

    results['detection'][backbone] = res

# ============================================================ autoencoder
# ae_scores.npy holds the reconstruction error per image, in h5 order
ae_scores = np.load(os.path.join(DATA_DIR, 'ae_scores.npy'))
ae_eval = ae_scores[test_idx]
mean_normal = ae_eval[y_eval == 0].mean()
mean_anomalous = ae_eval[y_eval == 1].mean()

results['autoencoder'] = {
    'auroc': float(roc_auc_score(y_eval, ae_eval)),
    'ap': float(average_precision_score(y_eval, ae_eval)),
    'mean_normal': float(mean_normal),
    'mean_anomalous': float(mean_anomalous),
    'gap_pct': float(100 * (mean_anomalous / mean_normal - 1)),
}
print('\n--- autoencoder (bottleneck 8) ---')
print('  AUROC %.4f   AP %.4f   normal %.6f   anomalous %.6f   gap %.1f%%' %
      (results['autoencoder']['auroc'], results['autoencoder']['ap'],
       mean_normal, mean_anomalous, results['autoencoder']['gap_pct']))


# ============================================================ confidence
results['confidence'] = {}
print('\n--- confidence (1 - max softmax) ---')
for backbone in ['classifier', 'classifier_noaug']:
    logits = load_features(backbone)['logits'][test_idx]
    scores = confidence_score(logits)
    confidence = 1 - scores
    results['confidence'][backbone] = {
        'auroc': float(roc_auc_score(y_eval, scores)),
        'ap': float(average_precision_score(y_eval, scores)),
        'mean_confidence_normal': float(confidence[y_eval == 0].mean()),
        'mean_confidence_anomalous': float(confidence[y_eval == 1].mean()),
    }
    print('  %-18s AUROC %.4f   AP %.4f   confidence: normal %.4f  anomalous %.4f' %
          (backbone, results['confidence'][backbone]['auroc'],
           results['confidence'][backbone]['ap'],
           confidence[y_eval == 0].mean(), confidence[y_eval == 1].mean()))

# ============================================================ classifier
results['classifier'] = {}
for backbone in ['classifier', 'classifier_noaug']:
    predictions = load_features(backbone)['logits'].argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels[test_idx], predictions[test_idx], labels=range(10))

    per_class = []
    for i in range(10):
        per_class.append({
            'name': CLASS_NAMES[i], 'precision': float(precision[i]),
            'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i]),
        })
    results['classifier'][backbone] = {
        'accuracy_train': float(accuracy_score(labels[train_idx], predictions[train_idx])),
        'accuracy_test': float(accuracy_score(labels[test_idx], predictions[test_idx])),
        'accuracy_all': float(accuracy_score(labels, predictions)),
        'per_class': per_class,
    }
    print('\n--- classifier: %s (test split only) ---' % backbone)
    print('  train %.4f   test %.4f   all 17736 %.4f  <- the last one is NOT a '
          'result, it includes training data' %
          (results['classifier'][backbone]['accuracy_train'],
           results['classifier'][backbone]['accuracy_test'],
           results['classifier'][backbone]['accuracy_all']))
    print(classification_report(labels[test_idx], predictions[test_idx],
                                target_names=CLASS_NAMES, digits=2))

# ============================================================ scores per class
feats = load_features('classifier')['feats']
query = feats[test_idx]
scores_imbalanced = knn_score(ref_imbalanced(feats, labels, pool), query, K)
scores_balanced = knn_score(ref_balanced(feats, labels, pool, 0), query, K)
scores_full = knn_score(ref_full(feats, labels, pool), query, K)

results['per_class_knn'] = []
print('--- mean kNN score per class (test split) ---')
print('  %-24s %11s %9s %7s' % ('class', 'broken', 'balanced', 'full'))
for c in NORMAL_CLASSES + [0, 1]:
    mask = labels[test_idx] == c
    row = {
        'name': CLASS_NAMES[c], 'anomalous': c in [0, 1], 'n': int(mask.sum()),
        'imbalanced': float(scores_imbalanced[mask].mean()),
        'balanced': float(scores_balanced[mask].mean()),
        'full': float(scores_full[mask].mean()),
    }
    results['per_class_knn'].append(row)
    print('  %-24s %11.2f %9.2f %7.2f %s' %
          (CLASS_NAMES[c], row['imbalanced'], row['balanced'], row['full'],
           '  <- ANOMALOUS' if c in [0, 1] else ''))

# ============================================================ ROC curves for the figures
reference = ref_balanced(feats, labels, pool, 0)
pca = fit_pca(reference)

curves = [
    ('kNN (balanced reference)', knn_score(reference, query, K)),
    ('Isolation Forest + PCA(450)', isolation_forest_score(reference, query, 0, 450, pca)),
    ('Isolation Forest (raw 512-d)', isolation_forest_score(reference, query, 0)),
    ('Autoencoder', ae_eval),
    ('Softmax confidence', confidence_score(load_features('classifier')['logits'][test_idx])),
]

results['roc_curves'] = {}
for name, scores in curves:
    fpr, tpr, _ = roc_curve(y_eval, scores)
    step = max(1, len(fpr) // 400)   # thin it out so the json stays small
    results['roc_curves'][name] = {
        'fpr': list(fpr[::step]) + [1.0],
        'tpr': list(tpr[::step]) + [1.0],
        'auroc': float(roc_auc_score(y_eval, scores)),
        'ap': float(average_precision_score(y_eval, scores)),
    }

# ============================================================ choosing a threshold
# A threshold read off the test scores would be fitted to the evaluation set.
# Calibrate on a held-out 10% of normal train galaxies instead, never touching
# the test split.
rng = np.random.default_rng(0)
shuffled = rng.permutation(len(pool))
n_calibration = len(pool) // 10
calibration_idx = pool[shuffled[:n_calibration]]
fit_idx = pool[shuffled[n_calibration:]]

chosen = []
per_class = [fit_idx[labels[fit_idx] == c] for c in NORMAL_CLASSES]
k_per_class = min(len(idx) for idx in per_class)
for idx in per_class:
    chosen.append(rng.choice(idx, k_per_class, replace=False))
calibration_reference = feats[np.sort(np.concatenate(chosen))]

calibration_scores = knn_score(calibration_reference, feats[calibration_idx], K)
eval_scores = knn_score(calibration_reference, query, K)

results['operating_points'] = []
print('\n--- choosing a threshold (calibrated on normal train galaxies only) ---')
for percentile in [90, 95, 99]:
    threshold = float(np.percentile(calibration_scores, percentile))
    flagged = (eval_scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, flagged, average='binary', zero_division=0)
    results['operating_points'].append({
        'rule': 'p%d of normal train' % percentile, 'threshold': threshold,
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'flagged': int(flagged.sum()),
    })
    print('  p%d  threshold %.2f   precision %.3f   recall %.3f   F1 %.3f   flagged %d/%d' %
          (percentile, threshold, precision, recall, f1, flagged.sum(), len(flagged)))

flagged = (eval_scores > 7.0).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_eval, flagged, average='binary', zero_division=0)
results['operating_points'].append({
    'rule': 'fixed threshold 7.0', 'threshold': 7.0,
    'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
    'flagged': int(flagged.sum()),
})
print('  fixed threshold 7.0           precision %.3f   recall %.3f   F1 %.3f   '
      'flagged %d/%d' % (precision, recall, f1, flagged.sum(), len(flagged)))

out_path = os.path.join(DOCS_DIR, 'results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print('\nWrote: %s' % out_path)
