# 🫀 Real-Time Arrhythmia Detection using PPG and Machine Learning

A real-time physiological monitoring system that detects irregular heart rhythms using PPG signals from the MAX30102 sensor and a machine learning model trained on the MIT-BIH dataset.

---

## 📌 Overview

This project implements an end-to-end pipeline for arrhythmia detection:

Sensor → Signal Processing → Feature Extraction → ML Inference → Output

The system runs on a Raspberry Pi and provides real-time:

- Heart Rate (BPM)
- Blood Oxygen (SpO₂)
- Arrhythmia Probability
- Classification (Normal / Irregular / Arrhythmia Likely)

---

## ⚙️ Hardware

- Raspberry Pi  
- MAX30102 Pulse Oximeter Sensor  

---

## 🧠 Software Stack

- Python  
- NumPy, SciPy  
- scikit-learn  
- SMBus (I2C communication)

---

## 🔬 Signal Processing Pipeline

### 1. Data Acquisition
- IR and Red signals sampled at 50 Hz

### 2. Filtering
- Butterworth bandpass filter (0.5 – 4 Hz)
- Removes motion noise and baseline drift

### 3. Peak Detection
- Derivative-based method
- Ensures minimum gap between beats

### 4. RR Interval Extraction
- Time difference between consecutive peaks
- Represents heart rhythm variability

### 5. RR Cleaning
- Moving average smoothing
- Outlier removal using median filtering

---

## 🤖 Machine Learning

### Dataset
- MIT-BIH Arrhythmia Database (ECG-based, expert labeled)

### Features (HRV-based)
- Mean RR
- Standard deviation
- Variance
- RMSSD
- pNN50
- Min, Max, Median, Range

### Model
- Gradient Boosting Classifier

### Performance
- Arrhythmia Recall: **~85–90%**
- Balanced for **high sensitivity (safety-focused system)**

---

## 🧠 Real-World Enhancements

To handle real-world PPG signals:

- RR smoothing and cleaning  
- Threshold tuning (0.45–0.47)  
- Breathing detection (reduces false positives)  
- Probability smoothing (stable output)  

---

## 📊 Output

Example:
