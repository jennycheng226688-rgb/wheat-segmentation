import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
)

from data_loader import get_image_mask_pairs
from features import extract_features


X_train = np.load("../EWS-Dataset/X_train.npy")
y_train = np.load("../EWS-Dataset/y_train.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("The number of zeros:", (y_train == 0).sum())
print("The number of 1:", (y_train == 1).sum())



# Train the final model with the optimal parameters selected by validation
best_C = 1
best_gamma = 0.1

print("\nUse optimal parameters:")
print("C =", best_C)
print("gamma =", best_gamma)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
model.fit(X_train_scaled, y_train)

print("Final model training completed")



# rread test data
test_folder = "../EWS-Dataset/test"
test_images, test_masks = get_image_mask_pairs(test_folder)

print("\ntest picture quantity:", len(test_images))
print("number of test masks:", len(test_masks))


acc_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []

for i in range(len(test_images)):
    img_test = cv2.imread(test_images[i])
    mask_test = cv2.imread(test_masks[i], cv2.IMREAD_GRAYSCALE)
    mask_test = (mask_test > 0).astype("uint8")

    X_test = extract_features(img_test)
    y_test = mask_test.flatten()

    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    iou = jaccard_score(y_test, y_pred_test, zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    iou_list.append(iou)

    print(
        f"Test image {i+1}/{len(test_images)} -> "
        f"Accuracy: {acc:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, "
        f"IoU: {iou:.4f}"
    )



print("\n Final Average Test Metrics ")
print(f"Average Test Accuracy: {np.mean(acc_list):.4f}")
print(f"Average Test Precision: {np.mean(precision_list):.4f}")
print(f"Average Test Recall: {np.mean(recall_list):.4f}")
print(f"Average Test F1: {np.mean(f1_list):.4f}")
print(f"Average Test IoU: {np.mean(iou_list):.4f}")



img_test = cv2.imread(test_images[0])
mask_test = cv2.imread(test_masks[0], cv2.IMREAD_GRAYSCALE)
mask_test = (mask_test > 0).astype("uint8")

X_test = extract_features(img_test)
X_test_scaled = scaler.transform(X_test)
y_pred_test = model.predict(X_test_scaled)

pred_mask = y_pred_test.reshape(mask_test.shape).astype("uint8")
pred_mask_clean = cv2.medianBlur(pred_mask, 3)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Test Image")
plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask_test, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred_mask_clean, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
