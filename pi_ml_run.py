import time
import numpy as np
from smbus2 import SMBus
from scipy.signal import butter, filtfilt
import pickle
import csv

# Load model file
model = pickle.load(open("model_GB.pkl", "rb"))
threshold = 0.47

prob_history = []

# MAX30102 Config (from site)
I2C_ADDR = 0x57
REG_FIFO_DATA = 0x07
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D

bus = SMBus(1)

def max30102_init():
    bus.write_byte_data(I2C_ADDR, REG_MODE_CONFIG, 0x03)
    bus.write_byte_data(I2C_ADDR, REG_SPO2_CONFIG, 0x27)
    bus.write_byte_data(I2C_ADDR, REG_LED1_PA, 0x1F)
    bus.write_byte_data(I2C_ADDR, REG_LED2_PA, 0x1F)

# Functions
def read_sample():
    data = bus.read_i2c_block_data(I2C_ADDR, REG_FIFO_DATA, 6)

    red = (data[0]<<16 | data[1]<<8 | data[2]) & 0x03FFFF
    ir  = (data[3]<<16 | data[4]<<8 | data[5]) & 0x03FFFF

    return red, ir

def bandpass(signal, fs):
    low = 0.5 / (fs/2)
    high = 4.0 / (fs/2)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(signal, fs):
    derivative = np.diff(signal)

    peaks = []
    last_peak = -1000
    min_gap = int(0.4 * fs)

    for i in range(1, len(derivative)):
        if derivative[i-1] > 0 and derivative[i] <= 0:

            threshold_local = np.mean(signal) + 0.5 * np.std(signal)

            if signal[i] > threshold_local:
                if i - last_peak > min_gap:
                    peaks.append(i)
                    last_peak = i

    return peaks

def bpm_time(peaks, fs):
    if len(peaks) < 2:
        return 0
    intervals = np.diff(peaks) / fs
    return 60 / np.mean(intervals)

def bpm_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    mask = (freqs > 0.5) & (freqs < 4.0)
    if not np.any(mask):
        return 0

    peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
    return peak_freq * 60

def calculate_spo2(red, ir):
    red = np.array(red)
    ir = np.array(ir)

    dc_red = np.mean(red)
    dc_ir = np.mean(ir)

    ac_red = np.std(red)
    ac_ir = np.std(ir)

    if dc_red == 0 or dc_ir == 0:
        return 0

    R = (ac_red/dc_red) / (ac_ir/dc_ir)
    spo2 = 110 - 25 * R

    return np.clip(spo2, 0, 100)

def get_rr_intervals(peaks, fs):
    if len(peaks) < 2:
        return []
    return np.diff(peaks) / fs

def extract_features(rr):
    rr = np.array(rr)

    if len(rr) < 5:
        return None

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

# Process RR to get good curves
def smooth_rr(rr):
    if len(rr) < 3:
        return rr
    return np.convolve(rr, np.ones(3)/3, mode='valid')

def clean_rr(rr):
    rr = np.array(rr)
    if len(rr) == 0:
        return rr

    median = np.median(rr)
    return rr[(rr > 0.5 * median) & (rr < 1.5 * median)]

def main():
    last_print_time = 0
    last_finger_time = 0

    file = open("detection.csv", "w", newline="")
    writer = csv.writer(file)

    max30102_init()

    fs = 50
    buffer_size = 600

    red_buf = []
    ir_buf = []

    print("Place finger...")

    while True:
        red, ir = read_sample()

        red_buf.append(red)
        ir_buf.append(ir)

        if len(ir_buf) > buffer_size:
            red_buf = red_buf[-buffer_size:]
            ir_buf = ir_buf[-buffer_size:]

            ir_np = np.array(ir_buf)
            ir_filtered = bandpass(ir_np, fs)

            peaks = detect_peaks(ir_filtered, fs)

            bpm1 = bpm_time(peaks, fs)
            bpm2 = bpm_fft(ir_filtered, fs)

            bpm = (bpm1 + bpm2)/2 if bpm1 and bpm2 else bpm1 or bpm2

            if bpm < 40 or bpm > 180:
                bpm = 0

            spo2 = calculate_spo2(red_buf, ir_buf)

            # RR processing
            rr = get_rr_intervals(peaks, fs)
            rr = smooth_rr(rr)
            rr = clean_rr(rr)

            if len(rr) > 1:
                diff = np.abs(np.diff(rr))
                rr = rr[1:][diff < 0.25]

            # Breathing detection
            breathing_like = False
            if len(rr) > 5:
                if np.mean(np.abs(np.diff(rr))) < 0.05:
                    breathing_like = True

            # Predict probability using model
            window = 10
            if len(rr) >= window:
                segment = rr[-window:]
                features = extract_features(segment)
            else:
                features = None

            if features is not None:
                prob = model.predict_proba([features])[0][1]

                prob_history.append(prob)
                if len(prob_history) > 8:
                    prob_history.pop(0)

                avg_prob = sum(prob_history) / len(prob_history)

                # Print every 1 second
                current_time = time.time()
                if current_time - last_print_time > 1.0:
                    last_print_time = current_time
    

                    print("\n----------------------------")
                    print(f"BPM: {bpm:.1f} bpm")
                    print(f"SpO2: {spo2:.1f}%")
                    print(f"Arrhythmia Probability: {avg_prob:.2f}")

                    if avg_prob > 0.65 and not breathing_like:
                        status = "ARRHYTHMIA LIKELY"
                    elif avg_prob > 0.45:
                        status = "IRREGULAR (possible breathing)"
                    else:
                        status = "NORMAL"

                    print(f"Status: {status}")

                writer.writerow([
                    time.time(),
                    bpm,
                    spo2,
                    avg_prob,
                    status
                ])
            else:
                current_time = time.time()

                if current_time - last_finger_time > 0.5:
                    last_finger_time = current_time
                    print("Finger not placed correctly")

        time.sleep(1/fs)

# ------------------------
if __name__ == "__main__":
    main()