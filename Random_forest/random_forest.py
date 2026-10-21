import argparse
import os
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1-Score", "IoU"]


class WheatDataset:
    def __init__(self, base_path, subset="train", target_size=(350, 350)):
        self.data_dir = Path(base_path) / subset
        self.target_size = target_size

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset subset not found: {self.data_dir}")

        all_files = sorted(os.listdir(self.data_dir))
        self.image_names = [
            f
            for f in all_files
            if "_mask" not in f and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not self.image_names:
            raise ValueError(f"No images found in {self.data_dir}")

    def __len__(self):
        return len(self.image_names)

    def get_example(self, index):
        img_name = self.image_names[index]
        name_part, ext_part = os.path.splitext(img_name)
        mask_name = f"{name_part}_mask{ext_part}"

        img_path = self.data_dir / img_name
        mask_path = self.data_dir / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {img_name}: {mask_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = cv2.resize(mask, self.target_size)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.uint8)

        return img, mask


def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    img_float = image.astype(np.float32)
    r, g, b = cv2.split(img_float)
    exg = 2.0 * g - r - b

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.float32(np.sqrt(sobel_x**2 + sobel_y**2))

    exg = np.expand_dims(exg, axis=-1)
    gradient = np.expand_dims(gradient_magnitude, axis=-1)

    features = np.concatenate([image, hsv, exg, gradient], axis=-1)
    return features.reshape(-1, features.shape[-1])


def build_training_matrix(dataset, sample_rate=0.05):
    X_list, y_list = [], []

    for i in range(len(dataset)):
        img, mask = dataset.get_example(i)
        X_img = extract_features(img)
        y_img = (mask.reshape(-1) > 0).astype(np.uint8)

        num_samples = int(X_img.shape[0] * sample_rate)
        indices = np.random.choice(X_img.shape[0], num_samples, replace=False)

        X_list.append(X_img[indices])
        y_list.append(y_img[indices])

    return np.vstack(X_list), np.concatenate(y_list)


def calculate_metrics(predicted_mask, ground_truth_mask):
    pred = predicted_mask.astype(bool)
    target = ground_truth_mask.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    tp = intersection
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    total = target.size

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = intersection / union if union > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "IoU": iou,
    }


def predict_mask(img, model, threshold):
    h, w, _ = img.shape
    features = extract_features(img)
    probs = model.predict_proba(features)[:, 1]
    return (probs >= threshold).astype(np.uint8).reshape(h, w)


def evaluate_single_image(img, gt_mask, model, threshold):
    pred_mask = predict_mask(img, model, threshold)
    return calculate_metrics(pred_mask, gt_mask)


def tune_model(X_train, y_train, val_set, n_estimators_options, depth_options, threshold_options):
    best_iou = -1.0
    best_bundle = {}

    print("Starting Grid Search on Validation Set...")

    for n_est in n_estimators_options:
        for depth in depth_options:
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_train, y_train)

            for threshold in threshold_options:
                val_ious = []

                for i in range(len(val_set)):
                    img, gt = val_set.get_example(i)
                    metrics = evaluate_single_image(img, gt, clf, threshold)
                    val_ious.append(metrics["IoU"])

                avg_val_iou = np.mean(val_ious)
                print(
                    "Config: "
                    f"n_est={n_est}, depth={depth}, threshold={threshold:.2f} "
                    f"-> Val IoU: {avg_val_iou:.4f}"
                )

                if avg_val_iou > best_iou:
                    best_iou = avg_val_iou
                    best_bundle = {
                        "model": clf,
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "threshold": threshold,
                        "validation_iou": avg_val_iou,
                    }

    print(
        "\nTuning Complete. Best Configuration: "
        f"{best_bundle['n_estimators']} trees, "
        f"depth {best_bundle['max_depth']}, "
        f"threshold {best_bundle['threshold']:.2f}, "
        f"validation IoU {best_bundle['validation_iou']:.4f}"
    )

    return best_bundle


def evaluate_test_set(test_set, model, threshold):
    metrics_sum = {metric_name: 0 for metric_name in METRIC_NAMES}

    for i in range(len(test_set)):
        img, gt = test_set.get_example(i)
        metrics = evaluate_single_image(img, gt, model, threshold)

        for key in metrics_sum:
            metrics_sum[key] += metrics[key]

    return {key: value / len(test_set) for key, value in metrics_sum.items()}


def show_random_examples(test_set, model, threshold, num_examples):
    num_test = len(test_set)
    random_indices = np.random.choice(num_test, size=min(num_examples, num_test), replace=False)

    for image_idx in random_indices:
        image, gt_mask = test_set.get_example(image_idx)
        pred_mask = predict_mask(image, model, threshold)
        metrics = calculate_metrics(pred_mask, gt_mask)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Input Image (Index: {image_idx})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(f"RF Prediction (Threshold: {threshold:.2f})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def save_test_images(test_set, model, threshold, output_dir, limit=0):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_images = len(test_set) if limit <= 0 else min(limit, len(test_set))

    for image_idx in range(num_images):
        image, gt_mask = test_set.get_example(image_idx)
        pred_mask = predict_mask(image, model, threshold)
        metrics = calculate_metrics(pred_mask, gt_mask)

        mask_path = output_path / f"test_{image_idx:03d}_prediction_mask.png"
        cv2.imwrite(str(mask_path), pred_mask * 255)

        comparison_path = output_path / f"test_{image_idx:03d}_comparison.png"
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Input Image (Index: {image_idx})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(
            f"RF Prediction | Acc: {metrics['Accuracy']:.4f}, "
            f"IoU: {metrics['IoU']:.4f}, "
            f"F1: {metrics['F1-Score']:.4f}"
        )
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest pixel-wise wheat segmentation.")
    parser.add_argument("--data-dir", default="./EWS-Dataset", help="Path to EWS-Dataset.")
    parser.add_argument("--target-size", type=int, default=350, help="Image resize size.")
    parser.add_argument("--sample-rate", type=float, default=0.05, help="Pixel sample rate per train image.")
    parser.add_argument("--n-estimators", default="50,100", help="Comma-separated tree counts.")
    parser.add_argument("--max-depths", default="10,20", help="Comma-separated max depths.")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated probability thresholds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="NumPy random seed.")
    parser.add_argument("--model-out", default="rf_optimized_pipeline.pkl", help="Output model path.")
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Number of random qualitative examples to display after evaluation.",
    )
    parser.add_argument(
        "--save-test-images",
        default=None,
        help="Directory for saving test prediction masks and comparison images.",
    )
    parser.add_argument(
        "--save-limit",
        type=int,
        default=0,
        help="Maximum number of test images to save. Use 0 to save all test images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    target_size = (args.target_size, args.target_size)
    n_estimators_options = parse_int_list(args.n_estimators)
    depth_options = parse_int_list(args.max_depths)
    threshold_options = parse_float_list(args.thresholds)

    train_set = WheatDataset(base_path=args.data_dir, subset="train", target_size=target_size)
    val_set = WheatDataset(base_path=args.data_dir, subset="validation", target_size=target_size)
    test_set = WheatDataset(base_path=args.data_dir, subset="test", target_size=target_size)

    print(f"Dataset loaded: Train({len(train_set)}), Validation({len(val_set)}), Test({len(test_set)})")

    X_train, y_train = build_training_matrix(train_set, sample_rate=args.sample_rate)
    print(
        "Training matrix built. "
        f"Samples={X_train.shape[0]}, "
        f"features_per_pixel={X_train.shape[1]}, "
        f"labels={y_train.shape[0]}"
    )

    best_bundle = tune_model(
        X_train,
        y_train,
        val_set,
        n_estimators_options,
        depth_options,
        threshold_options,
    )

    final_model = best_bundle["model"]
    final_threshold = best_bundle["threshold"]

    print("\n--- Final Evaluation on Test Set ---")
    test_metrics = evaluate_test_set(test_set, final_model, final_threshold)

    print("Final Test Metrics (Average over all test images):")
    for key, value in test_metrics.items():
        print(f"- {key}: {value:.4f}")

    joblib.dump(
        {
            "model": final_model,
            "threshold": final_threshold,
            "target_size": target_size,
            "features": ["RGB", "HSV", "ExG", "Sobel gradient magnitude"],
        },
        args.model_out,
    )

    if args.show_examples > 0:
        show_random_examples(test_set, final_model, final_threshold, args.show_examples)

    if args.save_test_images is not None:
        save_test_images(
            test_set,
            final_model,
            final_threshold,
            args.save_test_images,
            args.save_limit,
        )


if __name__ == "__main__":
    main()
