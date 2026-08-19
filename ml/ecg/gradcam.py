import torch
import numpy as np
import matplotlib.pyplot as plt

from model import ECGCNN


MODEL_PATH = "ml/ecg/ecg_cnn.pth"
DATA_PATH = "data/ecg/processed_dataset.pt"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load model
model = ECGCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# Store activations and gradients
activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Last convolution layer
target_layer = model.features[8]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# Load data
data = torch.load(
    DATA_PATH,
    weights_only=True
)

X = data["X"]
y = data["y"]


# Find an abnormal ECG
idx = int(torch.where(y == 1)[0][0])

sample = X[idx].unsqueeze(0).unsqueeze(0)
sample = sample.to(device)


# Forward
output = model(sample)

prediction = torch.argmax(
    output,
    dim=1
).item()


probability = torch.softmax(
    output,
    dim=1
)[0, 1].item()


# Backward
model.zero_grad()

output[0, prediction].backward()


# Grad-CAM
weights = gradients.mean(
    dim=2,
    keepdim=True
)

cam = (
    weights * activations
).sum(dim=1)

cam = torch.relu(cam)

cam = cam.squeeze().detach().cpu().numpy()

# Resize CAM to ECG length
cam = np.interp(
    np.linspace(0, len(cam) - 1, 1800),
    np.arange(len(cam)),
    cam
)

cam = (
    cam - cam.min()
) / (
    cam.max() - cam.min() + 1e-8
)


# ECG signal
ecg = X[idx].numpy()


# Plot
plt.figure(figsize=(14, 5))

plt.plot(
    ecg,
    linewidth=1
)

plt.imshow(
    cam[np.newaxis, :],
    aspect="auto",
    extent=[0, 1800, ecg.min(), ecg.max()],
    alpha=0.45
)

plt.title(
    f"ECG Grad-CAM | Prediction: "
    f"{'Abnormal' if prediction == 1 else 'Normal'} | "
    f"Probability: {probability:.2f}"
)

plt.xlabel("ECG Samples")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    "ml/ecg/ecg_gradcam.png",
    dpi=150
)

plt.show()

print("Grad-CAM saved:")
print("ml/ecg/ecg_gradcam.png")