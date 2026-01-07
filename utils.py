import numpy as np
from scipy.signal import find_peaks

def extract_hrv_features(ecg_signal, fs=360):
    peaks, _ = find_peaks(ecg_signal, distance=fs*0.6)

    if len(peaks) < 3:
        return None

    rr = np.diff(peaks) / fs * 1000  # RR in ms

    mean_rr = np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    nn50 = np.sum(np.abs(np.diff(rr)) > 50)
    pnn50 = (nn50 / len(rr)) * 100

    return [mean_rr, sdnn, rmssd, nn50, pnn50]
