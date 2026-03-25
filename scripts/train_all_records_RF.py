import wfdb
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# ------------------------
# FEATURE EXTRACTION
# ------------------------

def extract_features(rr):
    rr = np.array(rr)
    diff_rr = np.diff(rr)

    if len(diff_rr) == 0:
        return None

    return [
        np.mean(rr),
        np.std(rr),
        np.var(rr),
        np.mean(diff_rr),
        np.sqrt(np.mean(diff_rr**2)),
        np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr),
        np.min(rr),
        np.max(rr),
        np.median(rr),
        np.ptp(rr)
    ]

# ------------------------
# LABEL SIMPLIFICATION
# ------------------------

def simplify(label):
    normal = ['N', 'L', 'R', 'e', 'j']
    return 0 if label in normal else 1

# ------------------------
# LOAD DATASET
# ------------------------

data_path = './data/mitdb'

records = sorted(set(
    f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')
))

X = []
y = []

for rec in records:
    print("Processing:", rec)

    record = wfdb.rdrecord(f'{data_path}/{rec}')
    annotation = wfdb.rdann(f'{data_path}/{rec}', 'atr')

    r_peaks = annotation.sample
    labels = annotation.symbol

    # Remove '+' annotations
    valid_indices = [i for i, l in enumerate(labels) if l != '+']
    r_peaks = r_peaks[valid_indices]
    labels = [labels[i] for i in valid_indices]

    # RR intervals
    rr_intervals = np.diff(r_peaks) / 360
    labels = labels[1:]

    # Sliding window
    window = 20

    for i in range(len(rr_intervals) - window):
        segment = rr_intervals[i:i+window]
        feat = extract_features(segment)

        if feat is not None:
            X.append(feat)
            y.append(simplify(labels[i+window]))

print("\nTotal samples:", len(X))

# ------------------------
# TRAIN TEST SPLIT
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# MODEL
# ------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------
# PREDICTION (PROBABILITY)
# ------------------------

y_probs = model.predict_proba(X_test)[:, 1]

# ------------------------
# THRESHOLD SWEEP
# ------------------------

best_threshold = 0.4   # or try 0.45

y_pred = (y_probs > best_threshold).astype(int)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
print("\nFinal Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------
# SAVE MODEL + THRESHOLD
# ------------------------

os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("\nModel + threshold saved.")