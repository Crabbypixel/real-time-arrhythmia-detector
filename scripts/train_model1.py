import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ------------------------
# LOAD DATA
# ------------------------

record = wfdb.rdrecord('./data/mitdb/100')
annotation = wfdb.rdann('./data/mitdb/100', 'atr')

signal = record.p_signal[:, 0]
r_peaks = annotation.sample
labels = annotation.symbol

# ------------------------
# CLEAN LABELS (remove '+')
# ------------------------

valid_indices = [i for i, l in enumerate(labels) if l != '+']

r_peaks = r_peaks[valid_indices]
labels = [labels[i] for i in valid_indices]

# ------------------------
# RR INTERVALS
# ------------------------

fs = 360
rr_intervals = np.diff(r_peaks) / fs
labels = labels[1:]  # align

# ------------------------
# FEATURE EXTRACTION
# ------------------------

def extract_features(rr):
    rr = np.array(rr)
    if len(rr) < 5:
        return None

    diff_rr = np.diff(rr)

    return [
        np.mean(rr),
        np.std(rr),
        np.sqrt(np.mean(diff_rr**2)),
        np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)
    ]

# ------------------------
# BUILD DATASET
# ------------------------

X = []
y = []

window = 10

for i in range(len(rr_intervals) - window):
    segment = rr_intervals[i:i+window]
    feat = extract_features(segment)

    if feat is not None:
        X.append(feat)
        y.append(labels[i+window])

# ------------------------
# SIMPLIFY LABELS
# ------------------------

def simplify(label):
    normal = ['N', 'L', 'R', 'e', 'j']

    if label in normal:
        return 0  # normal
    else:
        return 1  # arrhythmia

y = [simplify(l) for l in y]

# ------------------------
# TRAIN MODEL
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Dataset size:", len(X))
print("Accuracy:", accuracy)

# ------------------------
# SAVE MODEL
# ------------------------

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/model.pkl")