import cv2
import numpy as np

def extract_features(img):
    img_float = img.astype(np.float32)

    B, G, R = cv2.split(img_float)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    ExG = 2 * G - R - B

    # Local statistics of grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_mean = cv2.blur(gray, (3, 3))

    local_mean_sq = cv2.blur(gray ** 2, (3, 3))
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    features = np.stack([
        R.flatten(),
        G.flatten(),
        B.flatten(),
        H.flatten(),
        S.flatten(),
        V.flatten(),
        ExG.flatten(),
        local_mean.flatten(),
        local_std.flatten()
    ], axis=1)

    return features