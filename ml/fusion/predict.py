import os
import gc
import threading

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import models, transforms

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

# Prevent multiple prediction requests from loading
# ClinicalBERT / ResNet / ECG simultaneously.
PREDICTION_LOCK = threading.Lock()


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

FUSION_MODEL = os.path.join(
    BASE_DIR,
    "ml",
    "fusion",
    "cardio_fusion.pth"
)

ECG_MODEL = os.path.join(
    BASE_DIR,
    "ml",
    "ecg",
    "ecg_cnn.pth"
)

NLP_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ============================================================
# GLOBAL MODELS
# ============================================================

ecg_model = None
nlp_model = None
tokenizer = None
cv_model = None
fusion_model = None


# ============================================================
# MEMORY CLEANUP
# ============================================================

def force_cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_ecg():

    global ecg_model

    ecg_model = None

    force_cleanup()

    print(">>> ECG MODEL RELEASED", flush=True)


def release_nlp():

    global tokenizer
    global nlp_model

    tokenizer = None
    nlp_model = None

    force_cleanup()

    print(">>> CLINICALBERT RELEASED", flush=True)


def release_cv():

    global cv_model

    cv_model = None

    force_cleanup()

    print(">>> RESNET18 RELEASED", flush=True)


def release_fusion():

    global fusion_model

    fusion_model = None

    force_cleanup()

    print(">>> FUSION MODEL RELEASED", flush=True)


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
# LOAD ECG
# ============================================================

def load_ecg_model():

    global ecg_model

    if ecg_model is None:

        print(">>> LOADING ECG CNN", flush=True)

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

        force_cleanup()

        print(">>> ECG CNN READY", flush=True)

    return ecg_model


# ============================================================
# LOAD CLINICAL BERT
# ============================================================

def load_nlp_model():

    global tokenizer
    global nlp_model

    if tokenizer is None:

        print(
            ">>> LOADING CLINICALBERT TOKENIZER",
            flush=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            NLP_MODEL
        )

    if nlp_model is None:

        print(
            ">>> LOADING CLINICALBERT",
            flush=True
        )

        model = AutoModel.from_pretrained(
            NLP_MODEL
        )

        model.to(DEVICE)
        model.eval()

        nlp_model = model

        force_cleanup()

        print(
            ">>> CLINICALBERT READY",
            flush=True
        )

    return tokenizer, nlp_model


# ============================================================
# LOAD RESNET
# ============================================================

def load_cv_model():

    global cv_model

    if cv_model is None:

        print(
            ">>> LOADING RESNET18",
            flush=True
        )

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        model.fc = nn.Identity()

        model.to(DEVICE)
        model.eval()

        cv_model = model

        force_cleanup()

        print(
            ">>> RESNET18 READY",
            flush=True
        )

    return cv_model


# ============================================================
# LOAD FUSION
# ============================================================

def load_fusion():

    global fusion_model

    if fusion_model is None:

        print(
            ">>> LOADING CARDIOFUSION",
            flush=True
        )

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

        fusion_model = model

        force_cleanup()

        print(
            ">>> CARDIOFUSION READY",
            flush=True
        )

    return fusion_model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# NLP FEATURES
# ============================================================

def get_nlp_features(text):

    global tokenizer
    global nlp_model

    if not text:

        text = (
            "No clinical information provided."
        )

    tokenizer, model = load_nlp_model()

    print(
        ">>> RUNNING CLINICAL NLP",
        flush=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        features = (
            outputs
            .last_hidden_state[:, 0, :]
            .clone()
            .detach()
        )

    del inputs
    del outputs

    release_nlp()

    print(
        ">>> NLP FEATURES:",
        tuple(features.shape),
        flush=True
    )

    return features


# ============================================================
# X-RAY FEATURES
# ============================================================

def get_cv_features(image_path):

    if not image_path:

        raise ValueError(
            "X-ray image is required."
        )

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )

    model = load_cv_model()

    print(
        ">>> RUNNING X-RAY ANALYSIS",
        flush=True
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.inference_mode():

        features = (
            model(tensor)
            .clone()
            .detach()
        )

    del tensor
    del image

    release_cv()

    print(
        ">>> X-RAY FEATURES:",
        tuple(features.shape),
        flush=True
    )

    return features


# ============================================================
# PREPARE ECG
# ============================================================

def prepare_ecg_signal(ecg_path):

    if not ecg_path:

        raise ValueError(
            "ECG file is required."
        )

    if not os.path.exists(ecg_path):

        raise FileNotFoundError(
            f"ECG file not found: {ecg_path}"
        )

    extension = os.path.splitext(
        ecg_path
    )[1].lower()

    if extension == ".pdf":

        raise ValueError(
            "PDF ECG inference is not enabled. "
            "Please upload ECG as JPG or PNG."
        )

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png"
    }:

        raise ValueError(
            "Unsupported ECG format. "
            "Please upload JPG or PNG."
        )

    image = Image.open(
        ecg_path
    ).convert("L")

    image = image.resize(
        (1000, 256)
    )

    tensor = transforms.ToTensor()(
        image
    )

    signal = tensor.mean(
        dim=0
    )

    signal = signal.mean(
        dim=0
    )

    signal = signal.float()

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

    signal = torch.nn.functional.interpolate(
        signal,
        size=1000,
        mode="linear",
        align_corners=False
    )

    del tensor
    del image

    return signal


# ============================================================
# ECG FEATURES
# ============================================================

def get_ecg_result(ecg_path):

    model = load_ecg_model()

    signal = prepare_ecg_signal(
        ecg_path
    )

    print(
        ">>> RUNNING ECG ANALYSIS",
        flush=True
    )

    with torch.inference_mode():

        feature_map = model.features(
            signal
        )

        ecg_features = (
            feature_map
            .squeeze(-1)
            .clone()
            .detach()
        )

        output = model.classifier(
            ecg_features
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        abnormal_probability = (
            probabilities[0, 1].item()
        )

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
            )
    }

    del signal
    del feature_map
    del output
    del probabilities

    release_ecg()

    print(
        ">>> ECG FEATURES:",
        tuple(ecg_features.shape),
        flush=True
    )

    print(
        ">>> ECG RISK:",
        f"{result['risk_percentage']:.2f}%",
        flush=True
    )

    return ecg_features, result


# ============================================================
# CARDIOFUSION PREDICTION
# ============================================================

def predict_fusion(
    clinical_text,
    image_path,
    ecg_path
):

    # --------------------------------------------------------
    # ONLY ONE PREDICTION AT A TIME
    # --------------------------------------------------------

    acquired = PREDICTION_LOCK.acquire(
        timeout=300
    )

    if not acquired:

        raise RuntimeError(
            "Prediction service is busy. "
            "Please try again."
        )

    nlp_features = None
    cv_features = None
    ecg_features = None
    output = None
    probabilities = None

    try:

        print(
            "",
            flush=True
        )

        print(
            "=" * 65,
            flush=True
        )

        print(
            "CARDIOPE-AI MULTIMODAL INFERENCE",
            flush=True
        )

        print(
            "=" * 65,
            flush=True
        )


        # ====================================================
        # 1. NLP
        # ====================================================

        print(
            ">>> STEP 1/4: CLINICAL NLP",
            flush=True
        )

        nlp_features = get_nlp_features(
            clinical_text
        )

        force_cleanup()


        # ====================================================
        # 2. X-RAY
        # ====================================================

        print(
            ">>> STEP 2/4: X-RAY",
            flush=True
        )

        cv_features = get_cv_features(
            image_path
        )

        force_cleanup()


        # ====================================================
        # 3. ECG
        # ====================================================

        print(
            ">>> STEP 3/4: ECG",
            flush=True
        )

        ecg_features, ecg_result = (
            get_ecg_result(
                ecg_path
            )
        )

        force_cleanup()


        # ====================================================
        # VALIDATE FEATURES
        # ====================================================

        print(
            ">>> VALIDATING FEATURES",
            flush=True
        )

        print(
            "ECG:",
            tuple(ecg_features.shape),
            flush=True
        )

        print(
            "NLP:",
            tuple(nlp_features.shape),
            flush=True
        )

        print(
            "CV:",
            tuple(cv_features.shape),
            flush=True
        )

        if ecg_features.shape != (1, 128):

            raise ValueError(
                f"Invalid ECG feature shape: "
                f"{ecg_features.shape}"
            )

        if nlp_features.shape != (1, 768):

            raise ValueError(
                f"Invalid NLP feature shape: "
                f"{nlp_features.shape}"
            )

        if cv_features.shape != (1, 512):

            raise ValueError(
                f"Invalid CV feature shape: "
                f"{cv_features.shape}"
            )


        # ====================================================
        # 4. FUSION
        # ====================================================

        print(
            ">>> STEP 4/4: CARDIOFUSION",
            flush=True
        )

        model = load_fusion()

        print(
            ">>> RUNNING FUSION NETWORK",
            flush=True
        )

        with torch.inference_mode():

            output = model(
                ecg_features,
                nlp_features,
                cv_features
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            abnormal_probability = (
                probabilities[0, 1].item()
            )


        # ====================================================
        # ASSESSMENT
        # ====================================================

        assessment = (

            "HIGH RISK"

            if abnormal_probability >= 0.5

            else

            "LOW RISK"
        )


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

            "assessment":
                assessment,

            "ecg_analysis":
                ecg_result
        }


        # ====================================================
        # RESULT
        # ====================================================

        print(
            "",
            flush=True
        )

        print(
            "=" * 65,
            flush=True
        )

        print(
            "CARDIOPE-AI RESULT",
            flush=True
        )

        print(
            "=" * 65,
            flush=True
        )

        print(
            "FINAL RISK:",
            f"{result['risk_percentage']:.2f}%",
            flush=True
        )

        print(
            "ASSESSMENT:",
            result["assessment"],
            flush=True
        )

        print(
            "ECG RISK:",
            f"{ecg_result['risk_percentage']:.2f}%",
            flush=True
        )

        print(
            "=" * 65,
            flush=True
        )

        return result


    except Exception as e:

        print(
            "!!! PREDICTION ERROR !!!",
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        raise


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

        release_fusion()

        force_cleanup()

        print(
            ">>> PREDICTION MEMORY CLEANUP COMPLETE",
            flush=True
        )

        PREDICTION_LOCK.release()