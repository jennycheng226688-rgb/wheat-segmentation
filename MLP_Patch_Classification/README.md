# Method 2.3 — MLP Patch-Based Pixel Classification

**COMP9517 Computer Vision | Group Project**
**Author:** z5524135 (Zhipeng GU)
**Slide Deck:** `MLP_Patch_Presentation.pptx`

---

## 1. Task & Positioning

Binary segmentation on the EWS (Early Wheat Sensing) dataset: classify each pixel as Plant or Soil.

This method reformulates segmentation as a per-pixel patch problem. Unlike SLIC + Random Forest, which relies on hand-crafted colour and texture descriptors, the MLP learns directly from raw normalized RGB patches. It falls between traditional ML and end-to-end deep learning — every step is documented, and we report numbers directly from the notebook.

---

## 2. Technical Pipeline

### 2.1 Data Pipeline

| Step | Detail |
|---|---|
| Image–mask pairing | Deterministic naming: each `RGB.png` matched to `*_mask.png` |
| Train set | 30 images, sampled at stride 4 |
| Test set | 48 images, dense prediction at full resolution |
| Train/test split | Fixed seed across all experiments |

### 2.2 Anti-Leakage Preprocessing

1. Normalise patch values to [0, 1].
2. `StandardScaler` fitted on training patches only.
3. Test patches use `transform` only — no test statistics in training.

### 2.3 MLP Architecture

```
Input:  17×17×3 patch = 867-dim vector
Hidden: Dense 512  (ReLU + BatchNorm + Dropout 0.3)
        Dense 256  (ReLU + BatchNorm + Dropout 0.3)
        Dense 128  (ReLU + BatchNorm)
Output: Dense 2    (Softmax → Plant / Soil)
```

### 2.4 Optimisation Setup

| Parameter | Value |
|---|---|
| Optimiser | Adam (lr = 1e-3, with decay) |
| Regularisation | L2 (alpha = 1e-4) |
| Batch size | 256 (512 in some notebook runs) |
| Epochs | 30 max + Early Stopping (patience = 10, val fraction = 0.1) |
| Sampling | Balanced: 15,000 patches / class |

We kept the same architecture and optimisation setup across all protocols, so any performance difference reflects experimental design choices rather than implementation drift.

---

## 3. Experimental Protocol

Both evaluation modes produce numbers directly from notebook printouts.

### 3.1 Single-Run Evaluation

Baseline on the fixed test set:

| Metric | Value |
|---|---|
| Accuracy | 0.9266 |
| F1-Score | 0.9448 |
| IoU (Jaccard) | 0.8954 |
| Precision | 0.9545 |
| Recall | 0.9353 |

### 3.2 Repeated-Run Evaluation

Five fixed seeds: `[1, 27, 512, 79, 2179]`

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Accuracy | 0.9207 | 0.0057 | 0.9119 | 0.9275 |
| F1-Score | 0.9408 | 0.0040 | 0.9346 | 0.9459 |
| IoU (Jaccard) | 0.8882 | 0.0072 | 0.8772 | 0.8974 |
| Precision | 0.9453 | 0.0117 | 0.9331 | 0.9622 |
| Recall | 0.9365 | 0.0098 | 0.9219 | 0.9469 |

Convergence across 5 runs:

- Epochs to converge: **21.0 ± 3.4**
- Best validation score: **0.95 ± 0.01**
- Training time: **99.8 ± 14.8 s**
- Inference time: **22.6 ± 2.8 s**

### 3.3 Patch-Size Ablation

| Patch Size | Input Dim | F1-Score | IoU |
|---|---:|---:|---:|
| 7×7 | 147 | 0.9352 | — |
| 13×13 | 507 | 0.9279 | 0.8655 |
| 17×17 | 867 | 0.9206 | 0.8529 |
| 25×25 | 1875 | 0.9115 | 0.8374 |

In this setup, 13×13 scored higher than 17×17. We still used 17×17 as our primary setting — more context does not always help, and the gap is small.

### 3.4 Qualitative Analysis

Per-image IoU on test set: `[0.974, 0.839, 0.807, 0.896]`, mean = **0.8787**.

The confusion matrix shows Precision consistently exceeds Recall. We take this as evidence that the model suppresses background noise well, but misses some fine-leaf structures when lighting is uneven.

---

## 4. Comparison with Other Methods

| Method | F1-Score | IoU |
|---|---:|---:|
| SLIC + Random Forest | 0.957 | 0.918 |
| U-Net (reference) | ≈ 0.950 | — |
| MLP Patch (single-run) | 0.9448 | 0.8954 |
| MLP Patch (5-run avg) | 0.9408 ± 0.0040 | 0.8882 ± 0.0072 |

The MLP falls slightly below SLIC + RF on both metrics. The gap makes sense — per-pixel classification without spatial weight sharing struggles at boundaries, which is exactly what fully convolutional methods fix.

---

## 5. Limitations

- **Patch size sensitivity:** Larger patches do not reliably help; the right size is task-dependent.
- **Per-pixel inference cost:** Dense prediction is expensive compared to CNNs with shared filters.
- **No spatial coherence:** Each pixel is predicted independently; boundary regions show inconsistencies.
- **Stepping stone:** This was always intended as a bridge toward FCN/U-Net-type architectures.

---

## 6. Reproducibility

All numbers come straight from notebook printouts. The protocol is fully documented in `MLP_Patch.ipynb`. Metric reports are accompanied by intervals; seeds and split are fixed; the notebook can be re-run to reproduce every result.

---

## 7. File Structure

```
Method_2.3_MLP_Patch/
├── MLP_Patch_Classification.ipynb            # Jupyter notebook (pipeline & experiments)
└── README.md                  # This file
```

---

## 8. Third-Party Libraries

The following libraries are used in this project. All are standard scientific Python packages available via pip or conda.

| Library | Version | Purpose | Source |
|---|---|---|---|
| `numpy` | — | Array operations, random number generation | pip/conda |
| `scikit-learn` | — | `MLPClassifier`, `StandardScaler`, metrics | pip/conda |
| `Pillow` (PIL) | — | Loading and processing images | pip/conda |
| `matplotlib` | — | Visualisation (confusion matrix, qualitative results) | pip/conda |
| `scipy` | — | `confusion_matrix` in some notebook runs | pip/conda |

**Code written by myself:**

- Patch extraction and dataset construction: `extract_patches()` function
- Image–mask pairing logic
- Experiment loop and multi-seed evaluation protocol
- Ablation study over patch sizes
- Qualitative analysis (error maps, per-image IoU)
- All results printing and reporting

---

## 9. How to Run

### 9.1 Environment Setup

Requires Python 3.8 or later.

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn Pillow matplotlib scipy jupyter
```

### 9.2 Dataset

Place the EWS dataset images in a directory. The code expects paired files:

- `*_RGB.png` — original image
- `*_mask.png` — binary mask (Plant = 255, Soil = 0)

Configure the data path in the notebook:

```python
DATA_ROOT = '/path/to/EWS/dataset'   # <- change this
```

### 9.3 Running the Notebook

```bash
jupyter notebook MLP_Patch.ipynb
```

The notebook is divided into numbered sections. Run them in order:

1. **Setup** — imports, path configuration, reproducibility seeds
2. **Data Loading** — image–mask pairing, train/test split
3. **Patch Extraction** — `extract_patches()`, class balancing, dataset construction
4. **Training** — `StandardScaler` (train-only), MLP training, loss curve
5. **Single-Run Evaluation** — metrics on fixed test set
6. **Repeated-Run Evaluation** — 5-seed statistical evaluation
7. **Patch-Size Ablation** — 7×7, 13×13, 17×17, 25×25 comparison
8. **Qualitative Analysis** — confusion matrix, error maps, per-image IoU

### 9.4 Expected Output

Running the full notebook produces:

- **Console printout**: per-epoch training loss, per-run metrics (F1, IoU, accuracy), summary statistics with mean ± std
- **Saved figures**:
  - `mlp_training_loss.png` — training loss curve
  - `mlp_confusion_matrix.png` — confusion matrix heatmap
  - `mlp_qualitative_results.png` — 4 test images: input / ground truth / prediction / error map

### 9.5 Reproducibility

All random seeds are fixed (`RANDOM_SEED = 42` for data split; `RUN_SEEDS = [1, 27, 512, 79, 2179]` for multi-run). Re-running the notebook with the same data should produce identical results.

---

## 10. Code Documentation

Key functions are documented with inline comments:

```python
def extract_patches(img, mask, patch_size=17, stride=1):
    """Extract centered patches around each valid pixel.

    Args:
        img: RGB image (H, W, 3)
        mask: binary mask (H, W) — 1=plant, 0=soil
        patch_size: side length of square patch (default 17)
        stride: sampling stride for training efficiency

    Returns:
        X: patch feature matrix (N, patch_size*patch_size*3)
        y: binary labels (N,)
    """
```

All MLP hyperparameters, data split decisions, and anti-leakage steps are documented directly in the notebook cells preceding the relevant code.
