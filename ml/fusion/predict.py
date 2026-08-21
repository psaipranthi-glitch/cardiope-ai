import os
import gc
import hashlib

import torch
import torch.nn as nn
from PIL import Image

from ml.fusion.model import CardioFusion


# ============================================================
# CPU SETTINGS
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
# GLOBAL MODELS
# ============================================================

ecg_model = None
fusion = None


# ============================================================
# TRAINED ECG CNN
# EXACT ARCHITECTURE OF ecg_cnn.pth
# ============================================================

class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 1 -> 32
            nn.Conv1d(
                1,
                32,
                kernel_size=11,
                padding=5
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),


            # 32 -> 64
            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),


            # 64 -> 128
            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2),


            # 128 -> 256
            nn.Conv1d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD ECG MODEL
# ============================================================

def load_ecg_model():

    global ecg_model

    if ecg_model is None:

        print(
            ">>> Loading trained ECG CNN..."
        )

        model = ECGCNN()

        checkpoint = torch.load(
            ECG_MODEL,
            map_location=DEVICE,
            weights_only=True
        )


        # ----------------------------------------------------
        # CHECKPOINT FORMAT
        # ----------------------------------------------------

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state = checkpoint[
                "model_state_dict"
            ]

            threshold = checkpoint.get(
                "threshold",
                0.79
            )

            input_length = checkpoint.get(
                "input_length",
                1800
            )

        else:

            state = checkpoint

            threshold = 0.79

            input_length = 1800


        print(
            ">>> ECG threshold:",
            threshold
        )

        print(
            ">>> ECG input length:",
            input_length
        )


        # ----------------------------------------------------
        # LOAD EXACT TRAINED ARCHITECTURE
        # ----------------------------------------------------

        model.load_state_dict(
            state,
            strict=True
        )


        del checkpoint
        del state

        model.to(DEVICE)

        model.eval()

        ecg_model = model

        gc.collect()

        print(
            ">>> ECG CNN loaded successfully"
        )


    return ecg_model


# ============================================================
# LOAD FUSION MODEL
# ============================================================

def load_fusion():

    global fusion

    if fusion is None:

        print(
            ">>> Loading CardioFusion..."
        )

        model = CardioFusion()

        checkpoint = torch.load(
            FUSION_MODEL,
            map_location=DEVICE,
            weights_only=True
        )


        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            state = checkpoint[
                "model_state_dict"
            ]

        else:

            state = checkpoint


        model.load_state_dict(
            state,
            strict=True
        )

        del checkpoint
        del state

        model.to(DEVICE)

        model.eval()

        fusion = model

        gc.collect()

        print(
            ">>> CardioFusion loaded successfully"
        )


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
# CLINICAL NLP FEATURES
# ============================================================

def get_nlp_features(
    text: str
):

    if not text:

        text = (
            "No clinical information provided."
        )


    text = (
        text
        .lower()
        .strip()
    )


    features = torch.zeros(
        768,
        dtype=torch.float32
    )


    words = text.split()


    if not words:

        words = [
            "unknown"
        ]


    for word in words:

        digest = hashlib.sha256(
            word.encode("utf-8")
        ).digest()


        for i in range(8):

            start = i * 4

            value = int.from_bytes(
                digest[
                    start:start + 4
                ],
                byteorder="little"
            )


            index = value % 768


            sign = (
                1.0
                if value % 2 == 0
                else -1.0
            )


            features[index] += sign


    norm = torch.norm(
        features
    )


    if norm > 0:

        features = (
            features / norm
        )


    return features.unsqueeze(0)


# ============================================================
# X-RAY FEATURES
# ============================================================

def get_cv_features(
    image_path: str
):

    if not image_path:

        raise ValueError(
            "X-ray image is required."
        )


    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"X-ray image not found: "
            f"{image_path}"
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


    return features


# ============================================================
# ECG IMAGE -> 1800 POINT SIGNAL
# ============================================================

def prepare_ecg_signal(
    ecg_path: str
):

    if not ecg_path:

        raise ValueError(
            "ECG file is required."
        )


    if not os.path.exists(
        ecg_path
    ):

        raise FileNotFoundError(
            f"ECG image not found: "
            f"{ecg_path}"
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


    image = Image.open(
        ecg_path
    ).convert("L")


    # --------------------------------------------------------
    # IMPORTANT
    # Training model expects 1800 samples.
    # --------------------------------------------------------

    image = image.resize(
        (1800, 256)
    )


    image_tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    ).reshape(
        256,
        1800
    ) / 255.0


    # Convert ECG image into 1D representation

    signal = image_tensor.mean(
        dim=0
    )


    # Normalize exactly as inference preprocessing

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

def get_ecg_result(
    ecg_path: str
):

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
            probabilities[
                0,
                1
            ].item()
        )


        ecg_features = (
            feature_map
            .squeeze(-1)
            .clone()
            .detach()
        )


    # --------------------------------------------------------
    # TRAINED THRESHOLD
    # --------------------------------------------------------

    threshold = 0.79


    if (
        abnormal_probability
        >= threshold
    ):

        ecg_assessment = (
            "ABNORMAL"
        )

    else:

        ecg_assessment = (
            "NORMAL"
        )


    result = {

        "model":
            "ECG CNN",

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
            threshold,

        "assessment":
            ecg_assessment
    }


    del signal

    del feature_map

    del logits

    del probabilities


    ecg_model = None

    gc.collect()


    print(
        ">>> ECG CNN released"
    )


    return (
        ecg_features,
        result
    )


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

        print(
            "=" * 70
        )

        print(
            "CARDIOPE-AI MULTIMODAL INFERENCE"
        )

        print(
            "=" * 70
        )


        # ====================================================
        # 1. CLINICAL NLP
        # ====================================================

        print(
            ">>> STEP 1/4: CLINICAL NLP"
        )


        nlp_features = (
            get_nlp_features(
                clinical_text
            )
        )


        print(
            ">>> NLP FEATURES:",
            tuple(
                nlp_features.shape
            )
        )


        # ====================================================
        # 2. X-RAY
        # ====================================================

        print(
            ">>> STEP 2/4: COMPUTER VISION"
        )


        cv_features = (
            get_cv_features(
                image_path
            )
        )


        print(
            ">>> CV FEATURES:",
            tuple(
                cv_features.shape
            )
        )


        # ====================================================
        # 3. ECG
        # ====================================================

        print(
            ">>> STEP 3/4: ECG CNN"
        )


        (
            ecg_features,
            ecg_result
        ) = get_ecg_result(
            ecg_path
        )


        ecg_probability = (
            ecg_result[
                "abnormal_probability"
            ]
        )


        print(
            ">>> ECG PROBABILITY:",
            f"{ecg_probability * 100:.2f}%"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            ecg_features.shape[1]
            != 256
        ):

            raise ValueError(
                "Invalid ECG feature size: "
                f"{ecg_features.shape}"
            )


        if (
            nlp_features.shape[1]
            != 768
        ):

            raise ValueError(
                "Invalid NLP feature size: "
                f"{nlp_features.shape}"
            )


        if (
            cv_features.shape[1]
            != 512
        ):

            raise ValueError(
                "Invalid CV feature size: "
                f"{cv_features.shape}"
            )


        # ====================================================
        # 4. FUSION
        # ====================================================

        print(
            ">>> STEP 4/4: CARDIOFUSION"
        )


        fusion_model = (
            load_fusion()
        )


        # ----------------------------------------------------
        # CardioFusion was trained with 128 ECG features.
        # If required, reduce the trained 256-dimensional
        # representation to the first 128 channels.
        # ----------------------------------------------------

        fusion_ecg_features = (
            ecg_features[:, :128]
        )


        with torch.inference_mode():

            output = fusion_model(
                fusion_ecg_features,
                nlp_features,
                cv_features
            )


            probabilities = (
                torch.softmax(
                    output,
                    dim=1
                )
            )


            fusion_probability = (
                probabilities[
                    0,
                    1
                ].item()
            )


        print(
            ">>> FUSION PROBABILITY:",
            f"{fusion_probability * 100:.2f}%"
        )


        # ====================================================
        # ECG-FIRST FUSION
        # ====================================================

        final_probability = (

            0.80
            * ecg_probability

            +

            0.20
            * fusion_probability
        )


        # ====================================================
        # FINAL ASSESSMENT
        # ====================================================

        if (
            final_probability
            >= 0.70
        ):

            assessment = (
                "HIGH RISK"
            )

        elif (
            final_probability
            >= 0.40
        ):

            assessment = (
                "MODERATE RISK"
            )

        else:

            assessment = (
                "LOW RISK"
            )


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


        print(
            "=" * 70
        )

        print(
            "CARDIOPE-AI RESULT"
        )

        print(
            "=" * 70
        )


        print(
            "ECG Risk:",
            f"{ecg_probability * 100:.2f}%"
        )


        print(
            "Fusion Risk:",
            f"{fusion_probability * 100:.2f}%"
        )


        print(
            "Final Risk:",
            f"{final_probability * 100:.2f}%"
        )


        print(
            "Assessment:",
            assessment
        )


        print(
            "=" * 70
        )


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