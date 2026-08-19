import torch
import sys
import os

# Make sure Python imports model.py from THIS folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ECGCNN


MODEL_PATH = "ml/ecg/ecg_cnn.pth"
DATA_PATH = "data/ecg/processed_dataset.pt"


print("Loading ECG model...")

model = ECGCNN()

print("Model architecture:")
print(model)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

print("ECG model loaded!")


# Load ECG dataset
data = torch.load(
    DATA_PATH,
    weights_only=False
)

X = data["X"]

print("ECG dataset:", X.shape)


# Add channel dimension
X = X.float().unsqueeze(1)


# Extract features
with torch.no_grad():

    features = model.features(X)

    features = features.squeeze(-1)


print("\nECG features generated")
print("Shape:", features.shape)
print("Feature size:", features.shape[-1])


# Save
torch.save(
    features,
    "data/ecg/ecg_features.pt"
)

print("\nSaved:")
print("data/ecg/ecg_features.pt")