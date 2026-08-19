import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import ECGCNN


# Load dataset
data = torch.load(
    "data/ecg/processed_dataset.pt",
    weights_only=True
)

X = data["X"]
y = data["y"]

print("Dataset:", X.shape)


# Add channel dimension
X = X.unsqueeze(1)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = ECGCNN().to(device)


# Class weights
normal = (y_train == 0).sum()
abnormal = (y_train == 1).sum()

weights = torch.tensor(
    [
        1.0,
        normal.float() / abnormal.float()
    ],
    dtype=torch.float32
).to(device)

print("Class weights:", weights)


criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        output = model(xb)

        loss = criterion(output, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss/len(train_loader):.4f}"
    )


# Evaluation
model.eval()

with torch.no_grad():

    X_test_device = X_test.to(device)

    outputs = model(X_test_device)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[:, 1].cpu().numpy()

    predictions = (
        probabilities >= 0.5
    ).astype(int)


y_true = y_test.numpy()


accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    probabilities
)


print("\n===== ECG CNN RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")


# Save model
torch.save(
    model.state_dict(),
    "ml/ecg/ecg_cnn.pth"
)

print("\nModel saved:")
print("ml/ecg/ecg_cnn.pth")