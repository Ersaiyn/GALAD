"""Gradio demo: classify a galaxy and compute its anomaly score.

Two things to note about how it is wired:

  * the reference set is built from the TRAIN split only, so a query is never
    compared against galaxies the reported evaluation treats as held out.
  * the threshold is read from docs/results.json, where it was calibrated on
    normal training galaxies rather than picked by looking at the test scores.

The classification and the anomaly score come from the same backbone, so high
confidence and a high anomaly score can coexist: the classifier has output
neurons for Disturbed and Merging, so those classes are not out-of-distribution
for it. Softmax confidence is NOT an anomaly signal here.

Run:  python demo/app.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.common import (CLASS_NAMES, DOCS_DIR, MODELS_DIR, load_features,
                        load_labels, master_split, reference_pool)
from src.detectors import knn_score, ref_balanced

K = 5

# take the threshold from docs/results.json so it cannot drift from the report
with open(os.path.join(DOCS_DIR, 'results.json')) as f:
    results = json.load(f)

threshold = None
precision = None
recall = None
for point in results['operating_points']:
    if point['rule'] == 'p95 of normal train':
        threshold = point['threshold']
        precision = point['precision']
        recall = point['recall']

device = torch.device('cpu')

model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 10)
model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, 'classifier.pth'), map_location=device))
last_layer = model.fc          # remember the last layer
model.fc = nn.Identity()       # now the model outputs 512 features
model.eval()

labels = load_labels()
train_idx, _ = master_split(labels)
pool = reference_pool(labels, train_idx)
feats = load_features('classifier')['feats']
reference = ref_balanced(feats, labels, pool, 0)

print('reference set: %d normal galaxies from the train split, threshold %.2f' %
      (len(reference), threshold))


def analyze(image):
    if image is None:
        return 'Upload a galaxy image.'

    image = image.convert('RGB').resize((256, 256))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.FloatTensor(array).unsqueeze(0)

    with torch.no_grad():
        features = model(tensor)
        logits = last_layer(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    score = knn_score(reference, features.numpy(), K)[0]
    verdict = 'ANOMALOUS' if score > threshold else 'normal'

    return (
        'Morphology:     %s\n'
        'Confidence:     %.1f%%\n'
        'Anomaly score:  %.2f   (threshold %.2f)\n'
        'Verdict:        %s\n\n'
        'At this threshold the detector runs at precision %.2f / recall %.2f\n'
        'on held-out data.\n'
        'The confidence above is the classifier\'s and is NOT an anomaly\n'
        'signal, see REPORT.md section 4.4.'
        % (CLASS_NAMES[predicted.item()], 100 * confidence.item(), score,
           threshold, verdict, precision, recall)
    )


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type='pil', label='Galaxy image'),
    outputs=gr.Textbox(label='Result', lines=9),
    title='GALAD - galaxy morphology anomaly detector',
    description=(
        'A ResNet-18 morphology classifier plus a kNN anomaly score in its '
        'feature space, trained on Galaxy10 DECaLS. The score measures distance '
        'from normal morphologies in a space that was itself learned from '
        'labels - read REPORT.md section 7 before applying this to other surveys.'
    ),
)

if __name__ == '__main__':
    demo.launch()
