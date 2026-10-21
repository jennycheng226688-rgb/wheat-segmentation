import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from methods.dlv3plus import DLv3Plus

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

pretrained = True  # Set to False to train from scratch

print('DeepLab v3+ ({})'.format('pretrained on ImageNet' if pretrained else 'from scratch'))
print('-' * 60)

model = DLv3Plus(pretrained=pretrained)

print('Training...')
train_metrics, val_metrics = model.train()
print('Train metrics:')
for k, v in train_metrics.items():
    print(f'  {k}: {v:.4f}')
print('Val metrics:')
for k, v in val_metrics.items():
    print(f'  {k}: {v:.4f}')

print('-' * 60)
print('Testing...')
test_metrics, preds, labels = model.test()
print('Test metrics:')
for k, v in test_metrics.items():
    print(f'  {k}: {v:.4f}')

# Per-metric per-class summary
print('\n{:<12} {:>10} {:>10} {:>10}'.format('Metric', 'Plants', 'Soil', 'Average'))
print('-' * 46)
for metric in ['f1', 'precision', 'recall', 'IoU']:
    v_plant = test_metrics.get(f'{metric} - class: plants', float('nan'))
    v_soil  = test_metrics.get(f'{metric} - class: soil',   float('nan'))
    print('{:<12} {:>10.4f} {:>10.4f} {:>10.4f}'.format(metric, v_plant, v_soil, (v_plant + v_soil) / 2))

# Visualization: Original / Ground Truth / Prediction
test_paths = sorted(glob.glob('data/test/*6.png'))
num_samples = min(4, len(test_paths))

fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
fig.suptitle('DeepLab v3+ — Test Results', fontsize=14)
col_titles = ['Original', 'Ground Truth', 'Prediction']

for i in range(num_samples):
    img = np.array(PILImage.open(test_paths[i]).convert('RGB'))
    gt  = labels[i].reshape(350, 350).numpy()
    pr  = preds[i].reshape(350, 350).numpy()

    for j, (data, cmap) in enumerate([(img, None), (gt, 'gray'), (pr, 'gray')]):
        axes[i, j].imshow(data, cmap=cmap)
        axes[i, j].set_title(col_titles[j])
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
print('\nVisualization saved to test_visualization.png')
