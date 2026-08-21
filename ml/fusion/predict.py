import os
import gc

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
#
# IMPORTANT:
# Only ONE heavy encoder is kept in memory at a time.
# ============================================================

ecg_model = None
tokenizer = None
nlp_model = None
cv_model = None
fusion = None


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    global ecg_model
    global tokenizer
    global nlp_model
    global cv_model
    global fusion

    ecg_model = None
    tokenizer = None
    nlp_model = None
    cv_model = None
    fusion = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
# LOAD ECG
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
# LOAD NLP
# ============================================================

def load_nlp_model():

    global tokenizer
    global nlp_model

    if tokenizer is None:

        print(">>> Loading ClinicalBERT tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            NLP_MODEL
        )

    if nlp_model is None:

        print(">>> Loading ClinicalBERT...")

        model = AutoModel.from_pretrained(
            NLP_MODEL
        )

        model.to(DEVICE)

        model.eval()

        nlp_model = model

        gc.collect()

        print(">>> ClinicalBERT loaded")

    return tokenizer, nlp_model


# ============================================================
# LOAD CV
# ============================================================

def load_cv_model():

    global cv_model

    if cv_model is None:

        print(">>> Loading ResNet18...")

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        model.fc = nn.Identity()

        model.to(DEVICE)

        model.eval()

        cv_model = model

        gc.collect()

        print(">>> ResNet18 loaded")

    return cv_model


# ============================================================
# LOAD FUSION
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

def get_nlp_features(text: str):

    global tokenizer
    global nlp_model

    tokenizer, model = load_nlp_model()

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

    # --------------------------------------------------------
    # CRITICAL:
    # Release ClinicalBERT immediately.
    # --------------------------------------------------------

    tokenizer = None
    nlp_model = None

    gc.collect()

    print(">>> ClinicalBERT released")

    return features


# ============================================================
# X-RAY FEATURES
# ============================================================

def get_cv_features(image_path: str):

    global cv_model

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

    # --------------------------------------------------------
    # CRITICAL:
    # Release ResNet18 immediately.
    # --------------------------------------------------------

    cv_model = None

    gc.collect()

    print(">>> ResNet18 released")

    return features


# ============================================================
# PREPARE ECG
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


    image_tensor = transforms.ToTensor()(
        image
    )


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


    del image_tensor
    del image

    return signal


# ============================================================
# ECG FEATURES
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

    # --------------------------------------------------------
    # CRITICAL:
    # Release ECG model.
    # --------------------------------------------------------

    ecg_model = None

    gc.collect()

    print(">>> ECG CNN released")


    return ecg_features, result


# ============================================================
# CARDIOFUSION PREDICTION
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
        # 1. NLP
        # ====================================================

        print()
        print("=" * 65)
        print("1/4 - CLINICAL NLP")
        print("=" * 65)

        nlp_features = get_nlp_features(
            clinical_text
        )

        print(
            "NLP:",
            tuple(nlp_features.shape)
        )

        gc.collect()


        # ====================================================
        # 2. X-RAY
        # ====================================================

        print()
        print("=" * 65)
        print("2/4 - X-RAY")
        print("=" * 65)

        cv_features = get_cv_features(
            image_path
        )

        print(
            "CV:",
            tuple(cv_features.shape)
        )

        gc.collect()


        # ====================================================
        # 3. ECG
        # ====================================================

        print()
        print("=" * 65)
        print("3/4 - ECG")
        print("=" * 65)

        ecg_features, ecg_result = get_ecg_result(
            ecg_path
        )

        print(
            "ECG:",
            tuple(ecg_features.shape)
        )

        print(
            "ECG Risk:",
            f"{ecg_result['risk_percentage']:.2f}%"
        )

        gc.collect()


        # ====================================================
        # VERIFY DIMENSIONS
        # ====================================================

        print()
        print("=" * 65)
        print("FEATURE VALIDATION")
        print("=" * 65)

        print(
            "ECG:",
            tuple(ecg_features.shape)
        )

        print(
            "NLP:",
            tuple(nlp_features.shape)
        )

        print(
            "CV:",
            tuple(cv_features.shape)
        )


        if ecg_features.shape[1] != 128:

            raise ValueError(
                f"Invalid ECG feature size: "
                f"{ecg_features.shape}"
            )


        if nlp_features.shape[1] != 768:

            raise ValueError(
                f"Invalid NLP feature size: "
                f"{nlp_features.shape}"
            )


        if cv_features.shape[1] != 512:

            raise ValueError(
                f"Invalid CV feature size: "
                f"{cv_features.shape}"
            )


        # ====================================================
        # 4. FUSION
        # ====================================================

        print()
        print("=" * 65)
        print("4/4 - CARDIOFUSION")
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

        print(
            "ECG Risk:",
            f"{ecg_result['risk_percentage']:.2f}%"
        )

        print("=" * 65)


        return result


    finally:

        # ====================================================
        # RELEASE REQUEST TENSORS
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


        # ====================================================
        # RELEASE ALL MODELS
        # ====================================================

        cleanup_memory()

        gc.collect()

        print(
            ">>> MEMORY CLEANUP COMPLETE"
        )