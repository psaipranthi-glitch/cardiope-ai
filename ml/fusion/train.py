import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import CardioFusion


# ==========================================
# 1. LOAD ECG FEATURES AND LABELS
# ==========================================

ecg_data = torch.load(
    "data/ecg/processed_dataset.pt",
    weights_only=True
)

ecg_features = torch.load(
    "data/ecg/ecg_features.pt",
    weights_only=True
)

y = ecg_data["y"]

print("ECG features:", ecg_features.shape)
print("Labels:", y.shape)


# ==========================================
# 2. LOAD NLP FEATURES
# ==========================================

nlp_features = torch.load(
    "data/nlp/clinical_features.pt",
    weights_only=True
)

print("NLP features:", nlp_features.shape)


# ==========================================
# 3. LOAD CV FEATURES
# ==========================================

cv_features = torch.load(
    "data/cv/cv_features.pt",
    weights_only=True
)

print("CV features:", cv_features.shape)


# ==========================================
# 4. ALIGN FEATURES
# ==========================================

# Currently we have one clinical description
# and one X-ray image.

# Repeat them for every ECG sample.

nlp_features = nlp_features.repeat(
    ecg_features.size(0),
    1
)

cv_features = cv_features.repeat(
    ecg_features.size(0),
    1
)


print("\nAfter alignment:")

print(
    "ECG:",
    ecg_features.shape
)

print(
    "NLP:",
    nlp_features.shape
)

print(
    "CV :",
    cv_features.shape
)


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

indices = torch.arange(
    len(y)
)

train_idx, test_idx = train_test_split(
    indices.numpy(),
    test_size=0.2,
    random_state=42,
    stratify=y.numpy()
)

train_idx = torch.tensor(
    train_idx,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_idx,
    dtype=torch.long
)


print("\nTraining samples:", len(train_idx))
print("Testing samples :", len(test_idx))


# ==========================================
# 6. CREATE MODEL
# ==========================================

model = CardioFusion()

print("\nFusion model created")


# ==========================================
# 7. CLASS WEIGHTS
# ==========================================

normal = (
    y[train_idx] == 0
).sum()

abnormal = (
    y[train_idx] == 1
).sum()


abnormal_weight = (
    normal.float() /
    abnormal.float()
)


weights = torch.tensor(
    [
        1.0,
        abnormal_weight
    ],
    dtype=torch.float32
)


print("\nClass distribution:")
print("Normal  :", normal.item())
print("Abnormal:", abnormal.item())

print(
    "Fusion class weights:",
    weights
)


# ==========================================
# 8. LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 9. TRAINING
# ==========================================

epochs = 10

print("\n===== FUSION TRAINING =====")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()


    output = model(
        ecg_features[train_idx],
        nlp_features[train_idx],
        cv_features[train_idx]
    )


    loss = criterion(
        output,
        y[train_idx]
    )


    loss.backward()

    optimizer.step()


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {loss.item():.4f}"
    )


# ==========================================
# 10. EVALUATION
# ==========================================

model.eval()


with torch.no_grad():

    output = model(
        ecg_features[test_idx],
        nlp_features[test_idx],
        cv_features[test_idx]
    )


    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1].numpy()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


# True labels

y_true = y[
    test_idx
].numpy()


# ==========================================
# 11. METRICS
# ==========================================

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


# ==========================================
# 12. RESULTS
# ==========================================

print(
    "\n===== CARDIOPE-AI FUSION RESULTS ====="
)

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


# ==========================================
# 13. SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "ml/fusion/cardio_fusion.pth"
)


print(
    "\nFusion model saved:"
)

print(
    "ml/fusion/cardio_fusion.pth"
)