import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def balanced_sample_pixels(X, y, n_samples=5000, random_state=42):
    np.random.seed(random_state)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n_half = n_samples // 2

    n0 = min(n_half, len(idx0))
    n1 = min(n_half, len(idx1))

    idx0_sample = np.random.choice(idx0, n0, replace=False)
    idx1_sample = np.random.choice(idx1, n1, replace=False)

    indices = np.concatenate([idx0_sample, idx1_sample])
    np.random.shuffle(indices)

    X_sample = X[indices]
    y_sample = y[indices]

    return X_sample, y_sample

def train_svm(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_scaled, y)

    return model, scaler

def predict_svm(model, scaler, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred






