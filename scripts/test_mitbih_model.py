import wfdb
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ------------------------
# LOAD MODEL
# ------------------------
model = pickle.load(open("model/model_GB.pkl", "rb"))
threshold = 0.45

# ------------------------
# FEATURE FUNCTION
# ------------------------
def extract_features(rr):
    rr = np.array(rr)
    diff_rr = np.diff(rr)

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
# LOAD RECORD
# ------------------------
record_name = "233"

record = wfdb.rdrecord(f"./data/mitdb/{record_name}")
annotation = wfdb.rdann(f"./data/mitdb/{record_name}", "atr")

r_peaks = annotation.sample
labels = annotation.symbol

valid_idx = [i for i, l in enumerate(labels) if l != '+']
r_peaks = r_peaks[valid_idx]
labels = [labels[i] for i in valid_idx]

rr = np.diff(r_peaks) / 360
labels = labels[1:]

# ------------------------
# EVALUATION
# ------------------------
window = 20

y_true = []
y_pred = []

normal = ['N', 'L', 'R', 'e', 'j']

for i in range(len(rr) - window):
    segment = rr[i:i+window]
    features = extract_features(segment)

    prob = model.predict_proba([features])[0][1]
    pred = 1 if prob > threshold else 0

    true_label = labels[i+window]
    true_class = 0 if true_label in normal else 1

    y_true.append(true_class)
    y_pred.append(pred)

# ------------------------
# RESULTS
# ------------------------
print(f"\nEvaluation on record {record_name}\n")

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))