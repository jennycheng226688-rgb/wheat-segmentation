## Method: Superpixel + Logistic Regression

**File:** `spplr.ipynb`

A traditional machine learning pipeline using SLIC superpixels and hand-crafted features.

### Dependencies

```bash
pip install numpy opencv-python scikit-image scikit-learn matplotlib
```

### How to Run

Open and run all cells in `spplr.ipynb`

### Pipeline

1. **Superpixel segmentation**: SLIC 
2. **Feature extraction**: 35 features total
   - Color statistics: RGB, HSV, LAB 
   - Vegetation indices: ExG, ExR, ExGR, CIVE, VEG
   - Texture: LBP histogram (10 bins)
   - Spatial: normalized centroid (x, y)
3. **Label assignment** 
4. **Classifier**: Logistic Regression with `class_weight="balanced"`, StandardScaler normalization
5. **Evaluation**: Precision, Recall, F1-score, Accuracy, Mean IoU


## Method : DeepLab v3+

**File:** `EWS-DLv3/run_dlv3plus.py`

A deep learning approach using DeepLab v3+ (ResNet-50 backbone) for binary semantic segmentation of wheat vs. soil.

Source code adapted from Zenkl et al. on github https://github.com/RadekZenkl/EWS

### Key Files

```
EWS-DLv3/
├── run_dlv3plus.py          # Main file
├── methods/
│   ├── base.py              # BenchmarkMethod base 
│   └── dlv3plus.py          # DLv3Plus model, training loop, CombinedLoss
├── helpers/
│   └── helper_fcts.py       # EWS_Dataset, data augmentation transforms
└── env.yml                  # Environment
```

### Dependencies


```bash
conda env create -f EWS-DLv3/env.yml
conda activate EWS
```

Requires PyTorch with CUDA. Verify GPU availability:

```python
import torch
print(torch.zeros(1).cuda())
```

The implementation uses `segmentation_models.pytorch`

### Data Setup

Place the dataset under `EWS-DLv3/data/`:

```
EWS-DLv3/data/
├── train/
├── validation/
└── test/
```

### Pipeline

1. **Data preprocessing**
2. **Model**:`DeepLabV3Plus` with ResNet-50 encoder
3. **CombinedLoss**: 0.5 × DiceLoss + 0.5 × CrossEntropyLoss
4. **Optimizer**: AdamW with CosineAnnealingLR
5. **Training**: best model checkpoint saved to `best_model.pth` 
6. **Evaluation**: per-image F1, Precision, Recall, IoU computed for both classes and averaged
7. **Visualization**: 4 test samples saved to `EWS-DLv3/test_visualization.png`

### How to Run

```bash
cd EWS-DLv3
python run_dlv3plus.py
```
---

## References

- Zenkl et al., *Outdoor Plant Segmentation with Deep Learning for High-Throughput Field Phenotyping on a Diverse Wheat Dataset*, Frontiers in Plant Science, 2021. https://doi.org/10.3389/fpls.2021.774068

- segmentation_models.pytorch: https://github.com/qubvel/segmentation_models.pytorch
