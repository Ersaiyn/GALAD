"""Draw every figure in REPORT.md from the numbers in docs/results.json.

Run evaluation/evaluate.py first.

Colours are written inline. Blue is normal, orange is anomalous, the same
everywhere. That pair stays distinguishable under colour-vision deficiency
and reads on a white background.

Run:  python evaluation/figures.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.common import (DATA_DIR, DOCS_DIR, FIGURES_DIR, is_anomalous, load_features,
                        load_labels, master_split, reference_pool)
from src.detectors import knn_score, ref_balanced

BLUE = '#2a78d6'       # normal
ORANGE = '#eb6834'     # anomalous
GREEN = '#1baf7a'
YELLOW = '#eda100'
PINK = '#e87ba4'
GREY = '#8a8984'

K = 5                  # same k as evaluation/evaluate.py

with open(os.path.join(DOCS_DIR, 'results.json')) as f:
    R = json.load(f)


# ------------------------------------------------------------ 1. ROC curves
plt.figure(figsize=(6.4, 5))
plt.plot([0, 1], [0, 1], color=GREY, linestyle='--', linewidth=1.2)
plt.text(0.6, 0.55, 'chance', color=GREY, fontsize=9, rotation=38)

for (name, curve), color in zip(R['roc_curves'].items(),
                                [BLUE, ORANGE, GREEN, YELLOW, PINK]):
    plt.plot(curve['fpr'], curve['tpr'], color=color, linewidth=2,
             label='%s — %.3f' % (name, curve['auroc']))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Detecting anomalous morphology on held-out galaxies')
plt.legend(loc='lower right', fontsize=8.5, title='AUROC', frameon=False)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_roc_curves.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig1_roc_curves.png')


# ------------------------------------------- 2. score distributions
labels = load_labels()
train_idx, test_idx = master_split(labels)
pool = reference_pool(labels, train_idx)
feats = load_features('classifier')['feats']
query = feats[test_idx]
is_anom = is_anomalous(labels[test_idx])

knn_scores = knn_score(ref_balanced(feats, labels, pool, 0), query, K)
ae_scores = np.load(os.path.join(DATA_DIR, 'ae_scores.npy'))[test_idx]

plt.figure(figsize=(11, 4.1))
panels = [
    (1, knn_scores, 'kNN: distance in feature space', 'kNN anomaly score',
     R['roc_curves']['kNN (balanced reference)']['auroc']),
    (2, ae_scores, 'Autoencoder: reconstruction error', 'mean squared error',
     R['autoencoder']['auroc']),
]
for position, scores, title, xlabel, auroc in panels:
    plt.subplot(1, 2, position)
    low = np.percentile(scores, 0.2)
    high = np.percentile(scores, 99.8)
    bins = np.linspace(low, high, 55)

    plt.hist(np.clip(scores[~is_anom], low, high), bins=bins, density=True,
             color=BLUE, alpha=0.6, label='Normal (classes 2-9)')
    plt.hist(np.clip(scores[is_anom], low, high), bins=bins, density=True,
             color=ORANGE, alpha=0.6, label='Anomalous (Disturbed, Merging)')

    plt.xlabel(xlabel)
    plt.ylabel('density')
    plt.title('%s\nAUROC %.3f' % (title, auroc), fontsize=11)
    plt.legend(fontsize=8.5, frameon=False)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_score_distributions.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig2_score_distributions.png')


# ------------------------------------------- 3. reference set composition
rows = R['per_class_knn'][::-1]        # reversed so the anomalous rows sit at the bottom
y = np.arange(len(rows))
imbalanced = [row['imbalanced'] for row in rows]
balanced = [row['balanced'] for row in rows]
names = [row['name'] + (' (anomalous)' if row['anomalous'] else '') for row in rows]

plt.figure(figsize=(7.6, 5))
for i in range(len(rows)):
    plt.plot([imbalanced[i], balanced[i]], [y[i], y[i]], color='#e6e5e1', linewidth=2)
plt.scatter(imbalanced, y, s=64, color=ORANGE, zorder=3,
            label='Two-class reference set')
plt.scatter(balanced, y, s=64, color=BLUE, zorder=3,
            label='Balanced reference set')

plt.yticks(y, names)
plt.xlabel('mean kNN score on held-out galaxies')
plt.title('With an unbalanced reference set, normal galaxies\n'
          'score higher than actual mergers', fontsize=11)
plt.legend(loc='upper right', fontsize=9, frameon=False)
plt.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_reference_set.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig3_reference_set.png')


# ------------------------------------------- 4. PCA component sweep
sweep = R['detection']['classifier']['if_pca']
components = sorted(int(k) for k in sweep)
auroc = [sweep[str(n)]['auroc'] for n in components]
variance = [sweep[str(n)]['cumulative_variance'] for n in components]
raw_auroc = R['detection']['classifier']['if_raw512']['auroc']

plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(components, auroc, color=BLUE, linewidth=2, marker='o', markersize=5)
plt.axhline(raw_auroc, color=GREY, linestyle='--', linewidth=1.2)
plt.text(60, raw_auroc + 0.012, 'no PCA, raw 512-d: %.3f' % raw_auroc,
         color='#52514e', fontsize=9)
plt.axhline(0.5, color=GREY, linestyle=':', linewidth=1)
plt.text(60, 0.512, 'chance', color=GREY, fontsize=9)
# the two points that refute the "PCA keeps 99% of the variance" explanation
plt.annotate('32 components\n%.3f' % auroc[components.index(32)],
             xy=(32, auroc[components.index(32)]), xytext=(80, 0.75), fontsize=8.5,
             arrowprops=dict(arrowstyle='-', color=GREY))
plt.annotate('450 components\n%.3f' % auroc[components.index(450)],
             xy=(450, auroc[components.index(450)]), xytext=(300, 0.775), fontsize=8.5,
             arrowprops=dict(arrowstyle='-', color=GREY))
plt.ylim(0.45, 0.95)
plt.xlabel('PCA components kept')
plt.ylabel('AUROC')
plt.title('Isolation Forest improves with more components', fontsize=11)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(components, variance, color=BLUE, linewidth=2, marker='o', markersize=5)
plt.annotate('32 components\n%.1f%%' % (100 * variance[components.index(32)]),
             xy=(32, variance[components.index(32)]), xytext=(80, 0.9958), fontsize=8.5,
             arrowprops=dict(arrowstyle='-', color=GREY))
plt.annotate('450 components\n%.1f%%' % (100 * variance[components.index(450)]),
             xy=(450, variance[components.index(450)]), xytext=(280, 0.9972), fontsize=8.5,
             arrowprops=dict(arrowstyle='-', color=GREY))
plt.ylim(0.990, 1.0008)
plt.xlabel('PCA components kept')
plt.ylabel('cumulative explained variance')
plt.title('...while retained variance barely moves', fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_pca_sweep.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig4_pca_sweep.png')


# ------------------------------------------- 5. t-SNE
rng = np.random.default_rng(42)
normal_test = test_idx[~is_anomalous(labels[test_idx])]
anomalous_test = test_idx[is_anomalous(labels[test_idx])]
sample = np.concatenate([rng.choice(normal_test, 700, replace=False),
                         rng.choice(anomalous_test, 400, replace=False)])

print('  running t-SNE on 1100 held-out galaxies...')
embedding = TSNE(n_components=2, perplexity=40, random_state=42,
                 max_iter=1000).fit_transform(feats[sample])
sample_anom = is_anomalous(labels[sample])
sample_scores = knn_score(ref_balanced(feats, labels, pool, 0), feats[sample], K)

plt.figure(figsize=(11.6, 5))

plt.subplot(1, 2, 1)
plt.scatter(embedding[~sample_anom, 0], embedding[~sample_anom, 1], s=13,
            color=BLUE, alpha=0.55, label='Normal (classes 2-9)')
plt.scatter(embedding[sample_anom, 0], embedding[sample_anom, 1], s=26,
            color=ORANGE, alpha=0.9, marker='^',
            label='Anomalous (Disturbed, Merging)')
plt.legend(loc='upper left', fontsize=9, frameon=False)
plt.title('Ground-truth labels', fontsize=11)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
low = np.percentile(sample_scores, 2)
high = np.percentile(sample_scores, 98)
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], s=15,
                      c=np.clip(sample_scores, low, high), cmap='Blues', alpha=0.9)
colorbar = plt.colorbar(scatter, pad=0.02, fraction=0.045)
colorbar.set_label('kNN anomaly score', fontsize=9)
plt.title('kNN score over the same map', fontsize=11)
plt.xticks([])
plt.yticks([])

plt.suptitle('ResNet-18 feature space, 512 -> 2 dimensions', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_tsne.png'), dpi=160, bbox_inches='tight')
plt.close()
print('  fig5_tsne.png')


# ------------------------------------------- 6. where the signal comes from
backbones = [
    ('Fine-tuned\n(augmented)', 'classifier'),
    ('Fine-tuned\n(no augmentation)', 'classifier_noaug'),
    ('ImageNet only\n(img/255)', 'imagenet'),
    ('ImageNet only\n(mean/std)', 'imagenet_norm'),
]
auroc = [R['detection'][key]['knn_balanced']['auroc'] for _, key in backbones]
ap = [R['detection'][key]['knn_balanced']['ap'] for _, key in backbones]
x = np.arange(len(backbones))
width = 0.36

plt.figure(figsize=(7.4, 4.3))
plt.bar(x - width / 2, auroc, width, color=BLUE, label='AUROC')
plt.bar(x + width / 2, ap, width, color=ORANGE, label='Average precision')
for i in range(len(backbones)):
    plt.text(x[i] - width / 2, auroc[i] + 0.016, '%.3f' % auroc[i],
             ha='center', fontsize=8.5)
    plt.text(x[i] + width / 2, ap[i] + 0.016, '%.3f' % ap[i], ha='center', fontsize=8.5)

plt.axhline(0.5, color=GREY, linestyle='--', linewidth=1.1,
            label='AUROC of a random detector - 0.500')
plt.axhline(R['protocol']['prevalence'], color=GREY, linestyle=':', linewidth=1.1,
            label='AP of a random detector - %.3f' % R['protocol']['prevalence'])

plt.xticks(x, [name for name, _ in backbones], fontsize=9)
plt.ylabel('score')
plt.ylim(0, 1.12)
plt.title('All of the detection signal comes from supervised fine-tuning\n'
          '(kNN k=5, balanced reference set, identical protocol)', fontsize=11)
plt.legend(loc='upper center', ncol=2, fontsize=8.5, frameon=False)
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'fig6_backbone_control.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig6_backbone_control.png')


# ------------------------------------------- 7. choosing a threshold
points = R['operating_points']
x = np.arange(len(points))
precision = [p['precision'] for p in points]
recall = [p['recall'] for p in points]

plt.figure(figsize=(6.6, 4.3))
plt.bar(x - width / 2, precision, width, color=BLUE, label='Precision')
plt.bar(x + width / 2, recall, width, color=ORANGE, label='Recall')
for i in range(len(points)):
    plt.text(x[i] - width / 2, precision[i] + 0.018, '%.2f' % precision[i],
             ha='center', fontsize=8.5)
    plt.text(x[i] + width / 2, recall[i] + 0.018, '%.2f' % recall[i],
             ha='center', fontsize=8.5)

plt.xticks(x, ['p90', 'p95', 'p99', 'fixed\n(7.0)'], fontsize=9)
plt.xlabel('threshold rule (percentile of normal training scores)')
plt.ylabel('score')
plt.ylim(0, 1.0)
plt.title('A high fixed threshold misses three quarters of anomalies', fontsize=11)
plt.legend(loc='upper left', fontsize=9, frameon=False)
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, 'fig7_operating_points.png'), dpi=160,
            bbox_inches='tight')
plt.close()
print('  fig7_operating_points.png')

print('Done.')
