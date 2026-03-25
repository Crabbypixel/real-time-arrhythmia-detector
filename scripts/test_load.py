import wfdb

record = wfdb.rdrecord('./data/mitdb/100')
annotation = wfdb.rdann('./data/mitdb/100', 'atr')

signal = record.p_signal[:, 0]
r_peaks = annotation.sample
labels = annotation.symbol

print("Signal length:", len(signal))
print("First 10 R-peaks:", r_peaks[:10])
print("First 10 labels:", labels[:10])

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(signal[:2000])
plt.scatter(r_peaks[:20], signal[r_peaks[:20]], color='red')
plt.title("ECG with R-peaks")
plt.show()

valid_indices = [i for i, l in enumerate(labels) if l != '+']

r_peaks = r_peaks[valid_indices]
labels = [labels[i] for i in valid_indices]

import numpy as np

fs = 360

rr_intervals = np.diff(r_peaks) / fs
labels = labels[1:]  # align labels

print("First 10 RR:", rr_intervals[:10])
print("First 10 labels:", labels[:10])