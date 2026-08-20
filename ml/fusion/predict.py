import os
import gc

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import models, transforms

from ml.fusion.model import CardioFusion


# ============================================================
# MEMORY / CPU SETTINGS
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
# LAZY MODEL REFERENCES
# ============================================================

fusion = None
ecg_model = None
tokenizer = None
nlp_model = None
cv_model = None


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

        x = self.classifier(x)

        return x


# ============================================================
# LOAD CARDIOFUSION
# ============================================================

def load_fusion():

    global fusion

    if fusion is None:

        print("Loading CardioFusion...")

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

        print("CardioFusion loaded!")

    return fusion


# ============================================================
# LOAD ECG MODEL
# ============================================================

def load_ecg_model():

    global ecg_model

    if ecg_model is None:

        print("Loading ECG CNN...")

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

        print("ECG CNN loaded!")

    return ecg_model


# ============================================================
# LOAD CLINICAL NLP
# ============================================================

def load_nlp_model():

    global tokenizer
    global nlp_model

    if tokenizer is None:

        print("Loading Clinical NLP tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            NLP_MODEL
        )

    if nlp_model is None:

        print("Loading Clinical NLP model...")

        model = AutoModel.from_pretrained(
            NLP_MODEL
        )

        model.to(DEVICE)

        model.eval()

        nlp_model = model

        gc.collect()

        print("Clinical NLP loaded!")

    return tokenizer, nlp_model


# ============================================================
# LOAD COMPUTER VISION MODEL
# ============================================================

def load_cv_model():

    global cv_model

    if cv_model is None:

        print("Loading CV model...")

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        model.fc = nn.Identity()

        model.to(DEVICE)

        model.eval()

        cv_model = model

        gc.collect()

        print("CV model loaded!")

    return cv_model


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
# CLINICAL NLP FEATURES
# ============================================================

def get_nlp_features(text: str):

    tokenizer, nlp_model = load_nlp_model()

    if not text:
        text = "No clinical information provided."

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=128
    )

    with torch.inference_mode():

        outputs = nlp_model(
            **inputs
        )

        features = (
            outputs
            .last_hidden_state[:, 0, :]
            .detach()
        )

    del inputs
    del outputs

    return features


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

    model = load_cv_model()

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.inference_mode():

        features = model(
            image
        ).detach()

    del image

    return features


# ============================================================
# PREPARE ECG SIGNAL
# ============================================================

def prepare_ecg_signal(ecg_path: str):

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
            "PDF ECG input is stored successfully, "
            "but PDF-to-signal inference is not enabled. "
            "Please upload the ECG as JPG or PNG."
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

    image_tensor = transforms.ToTensor(
    )(image)

    signal = image_tensor.mean(
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

    return signal


# ============================================================
# ECG FEATURES
# ============================================================

def get_ecg_features(ecg_path: str):

    signal = prepare_ecg_signal(
        ecg_path
    )

    model = load_ecg_model()

    with torch.inference_mode():

        features = model.features(
            signal
        )

        features = (
            features
            .squeeze(-1)
            .detach()
        )

    del signal

    return features


# ============================================================
# ECG PREDICTION
# ============================================================

def predict_ecg(ecg_path: str):

    if not ecg_path:
        return None

    extension = os.path.splitext(
        ecg_path
    )[1].lower()

    if extension == ".pdf":
        return None

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png"
    }:
        return None

    signal = prepare_ecg_signal(
        ecg_path
    )

    model = load_ecg_model()

    with torch.inference_mode():

        output = model(
            signal
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        abnormal_probability = (
            probabilities[0, 1].item()
        )

    del signal
    del output
    del probabilities

    return {

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


# ============================================================
# CARDIOFUSION
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

        # ====================================================
        # NLP
        # ====================================================

        print()
        print("=" * 65)
        print("GENERATING CLINICAL NLP FEATURES")
        print("=" * 65)

        nlp_features = get_nlp_features(
            clinical_text
        )

        print(
            "NLP features:",
            nlp_features.shape
        )


        # ====================================================
        # X-RAY
        # ====================================================

        print()
        print("=" * 65)
        print("GENERATING X-RAY FEATURES")
        print("=" * 65)

        cv_features = get_cv_features(
            image_path
        )

        print(
            "CV features:",
            cv_features.shape
        )


        # ====================================================
        # ECG
        # ====================================================

        print()
        print("=" * 65)
        print("GENERATING ECG FEATURES")
        print("=" * 65)

        ecg_features = get_ecg_features(
            ecg_path
        )

        print(
            "ECG features:",
            ecg_features.shape
        )


        # ====================================================
        # ECG PREDICTION
        # ====================================================

        print()
        print("=" * 65)
        print("RUNNING ECG PREDICTION")
        print("=" * 65)

        ecg_result = predict_ecg(
            ecg_path
        )


        # ====================================================
        # CARDIOFUSION
        # ====================================================

        print()
        print("=" * 65)
        print("RUNNING CARDIOFUSION")
        print("=" * 65)

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


        # ====================================================
        # RESULT
        # ====================================================

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
        # LOG
        # ====================================================

        print()
        print("=" * 65)
        print("CARDIOPE-AI RESULT")
        print("=" * 65)

        print(
            "Final Risk:",
            f"{result['risk_percentage']:.2f}%"
        )

        print(
            "Assessment:",
            result["assessment"]
        )

        if ecg_result:

            print(
                "ECG Risk:",
                f"{ecg_result['risk_percentage']:.2f}%"
            )

        print("=" * 65)

        return result


    finally:

        # ====================================================
        # RELEASE TEMPORARY TENSORS
        # ====================================================

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

        gc.collect()