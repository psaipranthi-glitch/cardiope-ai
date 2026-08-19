import os
import wfdb
import numpy as np
import torch

DATA_DIR = "data/ecg/raw"
OUTPUT = "data/ecg/processed_dataset.pt"

X = []
y = []

NORMAL = {"N"}
ABNORMAL = {"A", "V"}

# Find ECG records from .hea files
records = sorted([
    os.path.splitext(f)[0]
    for f in os.listdir(DATA_DIR)
    if f.endswith(".hea")
])

print("Records found:", records)

for record in records:

    path = os.path.join(DATA_DIR, record)

    try:
        signal, fields = wfdb.rdsamp(path)

        # First ECG lead: MLII
        signal = signal[:, 0]

        annotation = wfdb.rdann(path, "atr")

        count = 0

        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol
        ):

            if symbol in NORMAL:
                label = 0

            elif symbol in ABNORMAL:
                label = 1

            else:
                continue

            start = sample - 900
            end = sample + 900

            if start < 0 or end > len(signal):
                continue

            segment = signal[start:end]

            if len(segment) != 1800:
                continue

            # Normalize ECG segment
            segment = (
                segment - np.mean(segment)
            ) / (np.std(segment) + 1e-8)

            X.append(segment)
            y.append(label)

            count += 1

        print(f"Processing: {record} -> {count} beats")

    except Exception as e:
        print(f"Error processing {record}: {e}")


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("\nDataset created")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")
print("Normal  :", np.sum(y == 0))
print("Abnormal:", np.sum(y == 1))

os.makedirs("data/ecg", exist_ok=True)

torch.save(
    {
        "X": torch.tensor(X),
        "y": torch.tensor(y)
    },
    OUTPUT
)

print("\nSaved:")
print(OUTPUT)