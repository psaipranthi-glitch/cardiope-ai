import os
import gc
import hashlib

import torch
import torch.nn as nn
from PIL import Image

from ml.fusion.model import CardioFusion


# ============================================================
# CPU / MEMORY SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

ECG_MODEL = os.path.join(
    BASE_DIR,
    "ml",
    "ecg",
    "ecg_cnn.pth"
)

FUSION_MODEL = os.path.join(
    BASE_DIR,
    "ml",
    "fusion",
    "cardio_fusion.pth"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

ECG_SIGNAL_LENGTH = 1800

ECG_THRESHOLD = 0.79

ECG_WEIGHT = 0.80
FUSION_WEIGHT = 0.20


# ============================================================
# GLOBAL MODELS
# ============================================================

ecg_model = None
fusion = None


# ============================================================
# ECG CNN
# ============================================================

class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# LOAD ECG MODEL
# ============================================================

def load_ecg_model():

    global ecg_model

    if ecg_model is None:

        print(">>> Loading ECG CNN...")

        model = ECGCNN()

        state = torch.load(
            ECG_MODEL,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(state)

        del state

        model.to(DEVICE)

        model.eval()

        ecg_model = model

        gc.collect()

        print(">>> ECG CNN loaded")

    return ecg_model


# ============================================================
# LOAD FUSION MODEL
# ============================================================

def load_fusion():

    global fusion

    if fusion is None:

        print(">>> Loading CardioFusion...")

        model = CardioFusion()

        state = torch.load(
            FUSION_MODEL,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(state)

        del state

        model.to(DEVICE)

        model.eval()

        fusion = model

        gc.collect()

        print(">>> CardioFusion loaded")

    return fusion


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    global ecg_model
    global fusion

    ecg_model = None
    fusion = None

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# CLINICAL FEATURES
# ============================================================

def get_nlp_features(text: str):

    if not text:

        text = "No clinical information provided."

    text = text.lower().strip()

    features = torch.zeros(
        768,
        dtype=torch.float32
    )

    words = text.split()

    if not words:

        words = ["unknown"]

    for word in words:

        digest = hashlib.sha256(
            word.encode("utf-8")
        ).digest()

        for i in range(8):

            start = i * 4

            value = int.from_bytes(
                digest[start:start + 4],
                byteorder="little"
            )

            index = value % 768

            sign = (
                1.0
                if value % 2 == 0
                else -1.0
            )

            features[index] += sign

    norm = torch.norm(features)

    if norm > 0:

        features = features / norm

    return features.unsqueeze(0)


# ============================================================
# X-RAY FEATURES
# ============================================================

def get_cv_features(image_path: str):

    if not image_path:

        raise ValueError(
            "X-ray image is required."
        )

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (32, 16)
    )

    pixels = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    ) / 255.0

    features = pixels.repeat(
        1,
        512 // pixels.numel() + 1
    )[:, :512]

    mean = features.mean()

    std = features.std()

    features = (
        features - mean
    ) / (
        std + 1e-8
    )

    del pixels
    del image

    return features


# ============================================================
# ECG IMAGE -> 1800 POINT SIGNAL
# ============================================================

def prepare_ecg_signal(ecg_path: str):

    if not ecg_path:

        raise ValueError(
            "ECG file is required."
        )

    if not os.path.exists(ecg_path):

        raise FileNotFoundError(
            f"ECG image not found: {ecg_path}"
        )

    extension = os.path.splitext(
        ecg_path
    )[1].lower()

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png"
    }:

        raise ValueError(
            "Unsupported ECG format. "
            "Please upload JPG or PNG."
        )


    print(
        ">>> ECG preprocessing:"
        f" resizing to {ECG_SIGNAL_LENGTH} points"
    )


    image = Image.open(
        ecg_path
    ).convert("L")


    # IMPORTANT:
    # The training dataset contains 1800-point ECG signals.
    #
    # Therefore inference must also produce 1800 points.

    image = image.resize(
        (
            ECG_SIGNAL_LENGTH,
            256
        )
    )


    image_tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    ).reshape(
        256,
        ECG_SIGNAL_LENGTH
    ) / 255.0


    # Convert ECG image into a 1D waveform.

    signal = image_tensor.mean(
        dim=0
    )


    # Standardize exactly as a numerical signal.

    signal = (
        signal - signal.mean()
    ) / (
        signal.std() + 1e-8
    )


    signal = (
        signal
        .unsqueeze(0)
        .unsqueeze(0)
    )


    del image_tensor
    del image

    return signal


# ============================================================
# ECG PREDICTION
# ============================================================

def get_ecg_result(ecg_path: str):

    global ecg_model

    signal = prepare_ecg_signal(
        ecg_path
    )

    model = load_ecg_model()


    with torch.inference_mode():

        feature_map = model.features(
            signal
        )

        logits = model.classifier(
            feature_map
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        abnormal_probability = (
            probabilities[0, 1].item()
        )

        ecg_features = (
            feature_map
            .squeeze(-1)
            .clone()
            .detach()
        )


    # IMPORTANT:
    # This threshold came from ECG model evaluation.
    #
    # Do not use 0.40 or 0.50 here.

    ecg_positive = (
        abnormal_probability >= ECG_THRESHOLD
    )


    if ecg_positive:

        ecg_assessment = "ABNORMAL ECG"

    else:

        ecg_assessment = "NORMAL / LOWER-RISK ECG"


    result = {

        "abnormal_probability":
            round(
                abnormal_probability,
                4
            ),

        "risk_percentage":
            round(
                abnormal_probability * 100,
                2
            ),

        "threshold":
            ECG_THRESHOLD,

        "prediction":
            (
                1
                if ecg_positive
                else
                0
            ),

        "assessment":
            ecg_assessment
    }


    print(
        ">>> ECG probability:",
        f"{abnormal_probability * 100:.2f}%"
    )

    print(
        ">>> ECG threshold:",
        f"{ECG_THRESHOLD * 100:.0f}%"
    )

    print(
        ">>> ECG assessment:",
        ecg_assessment
    )


    del signal
    del feature_map
    del logits
    del probabilities


    ecg_model = None

    gc.collect()

    print(">>> ECG CNN released")


    return ecg_features, result


# ============================================================
# FINAL MULTIMODAL PREDICTION
# ============================================================

def predict_fusion(
    clinical_text: str,
    image_path: str,
    ecg_path: str
):

    nlp_features = None
    cv_features = None
    ecg_features = None
    output = None
    probabilities = None

    try:

        print()
        print("=" * 70)
        print("CARDIOPE-AI ECG-FIRST MULTIMODAL INFERENCE")
        print("=" * 70)


        # ====================================================
        # 1. CLINICAL
        # ====================================================

        print()
        print(">>> STEP 1/4: CLINICAL")

        nlp_features = get_nlp_features(
            clinical_text
        )

        print(
            ">>> NLP FEATURES:",
            tuple(nlp_features.shape)
        )


        # ====================================================
        # 2. X-RAY
        # ====================================================

        print()
        print(">>> STEP 2/4: X-RAY")

        cv_features = get_cv_features(
            image_path
        )

        print(
            ">>> CV FEATURES:",
            tuple(cv_features.shape)
        )


        # ====================================================
        # 3. ECG
        # ====================================================

        print()
        print(">>> STEP 3/4: ECG")

        ecg_features, ecg_result = get_ecg_result(
            ecg_path
        )

        ecg_probability = (
            ecg_result[
                "abnormal_probability"
            ]
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if ecg_features.shape[1] != 128:

            raise ValueError(
                "Invalid ECG feature size: "
                f"{ecg_features.shape}"
            )

        if nlp_features.shape[1] != 768:

            raise ValueError(
                "Invalid NLP feature size: "
                f"{nlp_features.shape}"
            )

        if cv_features.shape[1] != 512:

            raise ValueError(
                "Invalid CV feature size: "
                f"{cv_features.shape}"
            )


        # ====================================================
        # 4. FUSION SUPPORT
        # ====================================================

        print()
        print(">>> STEP 4/4: CARDIOFUSION")

        fusion_model = load_fusion()


        with torch.inference_mode():

            output = fusion_model(
                ecg_features,
                nlp_features,
                cv_features
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            fusion_probability = (
                probabilities[0, 1].item()
            )


        print(
            ">>> ECG RISK:",
            f"{ecg_probability * 100:.2f}%"
        )

        print(
            ">>> FUSION SUPPORT:",
            f"{fusion_probability * 100:.2f}%"
        )


        # ====================================================
        # ECG-FIRST FINAL SCORE
        # ====================================================

        final_probability = (

            ECG_WEIGHT * ecg_probability

            +

            FUSION_WEIGHT * fusion_probability
        )


        # ====================================================
        # IMPORTANT DECISION LOGIC
        # ====================================================
        #
        # ECG is the primary modality.
        #
        # If ECG crosses its validated threshold,
        # we do NOT allow the fusion model to downgrade
        # the patient to LOW RISK.
        #
        # This is especially important because the current
        # fusion training data repeats one NLP/X-ray sample
        # across all ECG records.
        # ====================================================

        if ecg_probability >= ECG_THRESHOLD:

            assessment = "HIGH RISK"

        elif final_probability >= 0.50:

            assessment = "MODERATE RISK"

        else:

            assessment = "LOW RISK"


        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "abnormal_probability":
                round(
                    final_probability,
                    4
                ),

            "risk_percentage":
                round(
                    final_probability * 100,
                    2
                ),

            "assessment":
                assessment,

            "ecg_analysis":
                ecg_result
        }


        # ====================================================
        # LOG
        # ====================================================

        print()
        print("=" * 70)
        print("CARDIOPE-AI RESULT")
        print("=" * 70)

        print(
            "ECG Probability:",
            f"{ecg_probability * 100:.2f}%"
        )

        print(
            "Fusion Support:",
            f"{fusion_probability * 100:.2f}%"
        )

        print(
            "Final Probability:",
            f"{final_probability * 100:.2f}%"
        )

        print(
            "ECG Threshold:",
            f"{ECG_THRESHOLD * 100:.0f}%"
        )

        print(
            "Assessment:",
            assessment
        )

        print("=" * 70)


        return result


    finally:

        if nlp_features is not None:

            del nlp_features

        if cv_features is not None:

            del cv_features

        if ecg_features is not None:

            del ecg_features

        if output is not None:

            del output

        if probabilities is not None:

            del probabilities


        cleanup_memory()

        gc.collect()

        print(
            ">>> MEMORY CLEANUP COMPLETE"
        )