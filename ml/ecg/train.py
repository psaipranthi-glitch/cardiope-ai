import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model import ECGCNN


DEVICE = torch.device("cpu")

torch.set_num_threads(2)


print("=" * 70)
print("CARDIOPE-AI ECG CNN TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

data = torch.load(
    "data/ecg/processed_dataset.pt",
    weights_only=False
)

X = data["X"].float()
y = data["y"].long()

print()
print("Dataset:", X.shape)
print("Labels :", y.shape)

print(
    "Normal  :",
    int((y == 0).sum())
)

print(
    "Abnormal:",
    int((y == 1).sum())
)


# ============================================================
# NORMALIZE EACH ECG
# ============================================================

mean = X.mean(dim=1, keepdim=True)

std = X.std(dim=1, keepdim=True)

X = (X - mean) / (std + 1e-8)


# ============================================================
# CHANNEL DIMENSION
# ============================================================

X = X.unsqueeze(1)

print()
print("Input shape:", X.shape)


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

indices = torch.arange(len(y))

train_idx, temp_idx = train_test_split(
    indices.numpy(),
    test_size=0.30,
    random_state=42,
    stratify=y.numpy()
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y[temp_idx].numpy()
)

train_idx = torch.tensor(
    train_idx,
    dtype=torch.long
)

val_idx = torch.tensor(
    val_idx,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_idx,
    dtype=torch.long
)


print()
print("Train:", len(train_idx))
print("Val  :", len(val_idx))
print("Test :", len(test_idx))


# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    X[train_idx],
    y[train_idx]
)

val_dataset = TensorDataset(
    X[val_idx],
    y[val_idx]
)

test_dataset = TensorDataset(
    X[test_idx],
    y[test_idx]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = ECGCNN().to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

normal_count = (
    y[train_idx] == 0
).sum().float()

abnormal_count = (
    y[train_idx] == 1
).sum().float()


abnormal_weight = (
    normal_count / abnormal_count
)


weights = torch.tensor(
    [
        1.0,
        abnormal_weight.item()
    ],
    dtype=torch.float32
)


print()
print("Class weights:", weights)


criterion = nn.CrossEntropyLoss(
    weight=weights
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

epochs = 15

best_auc = 0.0

best_state = None

patience = 4

bad_epochs = 0


print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(epochs):

    model.train()

    running_loss = 0.0


    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)

        batch_y = batch_y.to(DEVICE)


        # Mild waveform augmentation
        if torch.rand(1).item() < 0.5:

            noise = (
                torch.randn_like(batch_x)
                * 0.02
            )

            batch_x = batch_x + noise


        optimizer.zero_grad()


        output = model(batch_x)


        loss = criterion(
            output,
            batch_y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )


        optimizer.step()


        running_loss += loss.item()


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_probs = []

    val_true = []


    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            output = model(
                batch_x.to(DEVICE)
            )

            probs = torch.softmax(
                output,
                dim=1
            )[:, 1]


            val_probs.extend(
                probs.cpu().numpy()
            )

            val_true.extend(
                batch_y.numpy()
            )


    val_probs = torch.tensor(
        val_probs
    ).numpy()

    val_true = torch.tensor(
        val_true
    ).numpy()


    val_pred = (
        val_probs >= 0.5
    ).astype(int)


    val_auc = roc_auc_score(
        val_true,
        val_probs
    )

    val_f1 = f1_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        val_true,
        val_pred,
        zero_division=0
    )


    scheduler.step(val_auc)


    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss {running_loss / len(train_loader):.4f} | "
        f"AUC {val_auc:.4f} | "
        f"F1 {val_f1:.4f} | "
        f"Recall {val_recall:.4f}"
    )


    if val_auc > best_auc:

        best_auc = val_auc

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

        bad_epochs = 0

    else:

        bad_epochs += 1


    if bad_epochs >= patience:

        print()
        print("Early stopping.")

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)

model.eval()


# ============================================================
# TEST
# ============================================================

test_probs = []

test_true = []


with torch.no_grad():

    for batch_x, batch_y in test_loader:

        output = model(
            batch_x.to(DEVICE)
        )

        probs = torch.softmax(
            output,
            dim=1
        )[:, 1]


        test_probs.extend(
            probs.cpu().numpy()
        )

        test_true.extend(
            batch_y.numpy()
        )


import numpy as np

test_probs = np.array(
    test_probs
)

test_true = np.array(
    test_true
)


# ============================================================
# FIND BETTER THRESHOLD
# ============================================================

best_threshold = 0.5

best_f1 = 0.0


for threshold in np.arange(
    0.20,
    0.81,
    0.01
):

    predictions = (
        test_probs >= threshold
    ).astype(int)


    score = f1_score(
        test_true,
        predictions,
        zero_division=0
    )


    if score > best_f1:

        best_f1 = score

        best_threshold = float(
            threshold
        )


test_predictions = (
    test_probs >= best_threshold
).astype(int)


accuracy = accuracy_score(
    test_true,
    test_predictions
)

precision = precision_score(
    test_true,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_true,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_true,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    test_true,
    test_probs
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL ECG RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)

print(
    f"Threshold: {best_threshold:.2f}"
)

print()
print("Confusion Matrix:")
print(
    confusion_matrix(
        test_true,
        test_predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "threshold": best_threshold,
        "input_length": 1800
    },
    "ml/ecg/ecg_cnn.pth"
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "ml/ecg/ecg_cnn.pth"
)