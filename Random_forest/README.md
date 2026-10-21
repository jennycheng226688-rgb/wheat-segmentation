# Random Forest Wheat Segmentation

This repository contains a Random Forest baseline for pixel-wise wheat segmentation in the COMP9517 group project.

## Dataset

The code expects the EWS dataset in this structure:

```text
EWS-Dataset/
  train/
  validation/
  test/
```

Each image should have a corresponding binary mask with the `_mask` suffix, for example:

```text
sample.png
sample_mask.png
```

Images and masks are resized to `350 x 350` before feature extraction.

## Method

The segmentation task is treated as pixel-wise binary classification: wheat vs. background.

For each pixel, the model uses 8 handcrafted features:

- RGB channels
- HSV channels
- Excess Green index: `2G - R - B`
- Sobel gradient magnitude

To reduce computation, only `5%` of pixels from each training image are randomly sampled. A `RandomForestClassifier` is trained on the training set and selected using validation IoU.

## Model Tuning

| Parameter | Values |
| --- | --- |
| `n_estimators` | `50`, `100` |
| `max_depth` | `10`, `20` |
| threshold | `0.50`, `0.55`, `0.60`, `0.65`, `0.70` |

Best recorded configuration:

| Parameter | Value |
| --- | --- |
| `n_estimators` | `100` |
| `max_depth` | `10` |
| threshold | `0.50` |
| validation IoU | `0.8595` |

## Test Results

Average test-set performance from one recorded run:

| Metric | Score |
| --- | ---: |
| Accuracy | `0.9112` |
| Precision | `0.9118` |
| Recall | `0.9336` |
| F1-Score | `0.9194` |
| IoU | `0.8561` |

Small variations may occur because training pixels are randomly sampled.

## Requirements

Python 3.10 or later is recommended.

Required packages:

```bash
pip install opencv-python numpy scikit-learn matplotlib joblib jupyter
```

For `random_forest.ipynb`, open the notebook with Jupyter and select the same Python environment where these packages are installed.

## Run

```bash
python random_forest.py
```

Optional outputs:

```bash
python random_forest.py --save-test-images ./rf_test_outputs
```

This option automatically creates the result folder if it does not already exist.
The folder will contain prediction mask images and side-by-side comparison images
for the test set, for example:

```text
rf_test_outputs/
  test_000_prediction_mask.png
  test_000_comparison.png
```

The script prints dataset sizes, validation IoU for each configuration, and final test metrics. It also saves the best model and threshold to `rf_optimized_pipeline.pkl`.
