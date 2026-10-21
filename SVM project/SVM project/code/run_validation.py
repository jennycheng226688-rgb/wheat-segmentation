import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score

from data_loader import get_image_mask_pairs
from features import extract_features



#Read the saved train data
X_train = np.load("../EWS-Dataset/X_train.npy")
y_train = np.load("../EWS-Dataset/y_train.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("The number of zeros:", (y_train == 0).sum())
print("The number of 1:", (y_train == 1).sum())



# 
val_folder = "../EWS-Dataset/validation"
val_images, val_masks = get_image_mask_pairs(val_folder)

print("Number of validation images:", len(val_images))
print("Number of validation masks:", len(val_masks))


n_val_use = 5
print("Number of validation sheets used:", n_val_use)



# 3. Parameter tuning
C_list = [1, 10]
gamma_list = ["scale", 0.1]

best_iou = 0
best_C = None
best_gamma = None

for C in C_list:
    for gamma in gamma_list:
        print(f"\nTesting C={C}, gamma={gamma}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train_scaled, y_train)

        iou_list = []

        for i in range(n_val_use):
            img_val = cv2.imread(val_images[i])
            mask_val = cv2.imread(val_masks[i], cv2.IMREAD_GRAYSCALE)
            mask_val = (mask_val > 0).astype("uint8")

            X_val = extract_features(img_val)
            y_val = mask_val.flatten()

            X_val_scaled = scaler.transform(X_val)
            y_pred_val = model.predict(X_val_scaled)

            iou = jaccard_score(y_val, y_pred_val)
            iou_list.append(iou)

            print(f"  val image {i+1}/{n_val_use} IoU: {iou:.4f}")

        avg_iou = np.mean(iou_list)
        print("Average Validation IoU:", avg_iou)

        if avg_iou > best_iou:
            best_iou = avg_iou
            best_C = C
            best_gamma = gamma

print("\n Optimal parameters")
print("Best C:", best_C)
print("Best gamma:", best_gamma)
print("Best Average Validation IoU:", best_iou)