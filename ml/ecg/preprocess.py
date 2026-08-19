import wfdb
import numpy as np


def load_ecg(record_path):
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal[:, 0]
    fs = record.fs

    return signal, fs


def normalize(signal):
    signal = signal.astype(np.float32)

    mean = np.mean(signal)
    std = np.std(signal)

    return (signal - mean) / (std + 1e-8)


def create_segments(signal, segment_length=1800):
    segments = []

    for start in range(0, len(signal) - segment_length, segment_length):
        segment = signal[start:start + segment_length]
        segments.append(segment)

    return np.array(segments)


if __name__ == "__main__":

    signal, fs = load_ecg("data/ecg/raw/100")

    signal = normalize(signal)

    segments = create_segments(signal)

    print("ECG preprocessing successful")
    print("Sampling rate:", fs)
    print("Original signal:", signal.shape)
    print("Segments:", segments.shape)