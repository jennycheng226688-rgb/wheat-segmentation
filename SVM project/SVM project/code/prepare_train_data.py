import cv2
import numpy as np

from data_loader import get_image_mask_pairs
from features import extract_features
from train import balanced_sample_pixels



#
train_folder = "../EWS-Dataset/train"
train_images, train_masks = get_image_mask_pairs(train_folder)

print("Number of train pictures:", len(train_images))
print("Number of train masks:", len(train_masks))

samples_per_image = 200

X_list = []
y_list = []


# Process each image one by one
for i in range(len(train_images)):
    img = cv2.imread(train_images[i])
    mask = cv2.imread(train_masks[i], cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype("uint8")

    X = extract_features(img)
    y = mask.flatten()

    X_sample, y_sample = balanced_sample_pixels(
        X, y, n_samples=samples_per_image
    )

    X_list.append(X_sample)
    y_list.append(y_sample)

    print(f"train image {i+1}/{len(train_images)} done")


X_train = np.vstack(X_list)
y_train = np.hstack(y_list)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("The number of zeros:", (y_train == 0).sum())
print("The number of 1:", (y_train == 1).sum())


# save
np.save("../EWS-Dataset/X_train.npy", X_train)
np.save("../EWS-Dataset/y_train.npy", y_train)

print("Training data has been saved:")
print("../EWS-Dataset/X_train.npy")
print("../EWS-Dataset/y_train.npy")