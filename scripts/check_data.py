"""Print the dataset composition and the exact split every experiment uses.

Which classes count as anomalous is the central modelling decision of the
project, so it lives in src/common.py rather than in a pre-computed array, and
this script prints it along with the split and the reference pool.

Run:  python scripts/check_data.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.common import (ANOMALOUS_CLASSES, CLASS_NAMES, NORMAL_CLASSES,
                        is_anomalous, load_labels, master_split, reference_pool)

labels = load_labels()
train_idx, test_idx = master_split(labels)
pool = reference_pool(labels, train_idx)

print('Galaxy10 DECaLS: %d images, 256x256x3 (g, r, z bands)\n' % len(labels))
print('%2s %-24s %6s %6s %6s  role' % ('', 'class', 'total', 'train', 'test'))
for c in range(10):
    role = 'ANOMALOUS' if c in ANOMALOUS_CLASSES else 'normal'
    total = int((labels == c).sum())
    n_train = int((labels[train_idx] == c).sum())
    n_test = int((labels[test_idx] == c).sum())
    print('%2d %-24s %6d %6d %6d  %s' % (c, CLASS_NAMES[c], total, n_train, n_test, role))

n_anomalous = int(is_anomalous(labels).sum())
print('\nnormal (classes 2-9): %d    anomalous (classes 0,1): %d' %
      (len(labels) - n_anomalous, n_anomalous))

print('\nTHE SPLIT  80/20, stratified, random_state=42')
print('  train          %d' % len(train_idx))
print('  test           %d  = %d normal + %d anomalous' %
      (len(test_idx), int((~is_anomalous(labels[test_idx])).sum()),
       int(is_anomalous(labels[test_idx]).sum())))
print('  prevalence     %.4f   <- average precision of a random detector' %
      is_anomalous(labels[test_idx]).mean())

print('\nREFERENCE POOL  normal galaxies from the train split: %d' % len(pool))
per_class = {}
for c in NORMAL_CLASSES:
    per_class[c] = int((labels[pool] == c).sum())
print('  per class:', per_class)

cap = min(per_class.values())
print('  the rarest normal class caps a balanced reference set at %d x 8 = %d' %
      (cap, cap * 8))
would_get = sum(min(375, v) for v in per_class.values())
print('  asking for 375 per class would silently give %d, not 3000,'
      ' because Cigar Shaped only has %d' % (would_get, per_class[4]))
print('  so the balanced reference set is built by capping at the rarest class')

class_sorted = pool[np.argsort(labels[pool], kind='stable')][:3000]
print('\n  for comparison, the first 3000 of the CLASS-SORTED pool contain only:')
for c in NORMAL_CLASSES:
    count = int((labels[class_sorted] == c).sum())
    if count > 0:
        print('    %-24s %d' % (CLASS_NAMES[c], count))
print('  which is why that reference set scores at chance (see REPORT.md 4.3)')
