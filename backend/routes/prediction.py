import os
import uuid
import traceback

from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile
)

from backend.auth import get_current_user

from backend.database import (
    patients_collection,
    predictions_collection
)

from ml.fusion.predict import predict_fusion


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


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

XRAY_UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "data",
    "xray",
    "uploads"
)

ECG_UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "data",
    "ecg",
    "uploads"
)

os.makedirs(
    XRAY_UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    ECG_UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# SAVE UPLOAD
# ============================================================

async def save_uploaded_file(
    upload: UploadFile,
    directory: str,
    allowed_types: set,
    allowed_extensions: set,
    prefix: str
):

    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=f"{prefix.upper()} file is required."
        )

    content_type = (
        upload.content_type or ""
    ).lower()

    original_filename = (
        upload.filename or ""
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    print(
        f">>> RECEIVING {prefix.upper()}: "
        f"{original_filename} | "
        f"{content_type}",
        flush=True
    )

    if content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {prefix} file type: "
                f"{content_type}"
            )
        )

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {prefix} file extension: "
                f"{extension}"
            )
        )

    filename = (
        f"{prefix}_"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    file_path = os.path.join(
        directory,
        filename
    )

    try:

        contents = await upload.read()

    except Exception as e:

        print(
            f">>> ERROR READING {prefix.upper()} FILE:",
            repr(e),
            flush=True
        )

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read {prefix} file."
        )

    if not contents:

        raise HTTPException(
            status_code=400,
            detail=f"Uploaded {prefix} file is empty."
        )

    try:

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(contents)

    except Exception as e:

        print(
            f">>> ERROR SAVING {prefix.upper()} FILE:",
            repr(e),
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save {prefix} file."
        )

    print(
        f">>> {prefix.upper()} SAVED:",
        file_path,
        flush=True
    )

    print(
        f">>> {prefix.upper()} SIZE:",
        len(contents),
        "bytes",
        flush=True
    )

    return filename, file_path


# ============================================================
# CLEANUP
# ============================================================

def cleanup_file(path):

    try:

        if path and os.path.exists(path):

            os.remove(path)

            print(
                ">>> CLEANED:",
                path,
                flush=True
            )

    except Exception as e:

        print(
            ">>> CLEANUP ERROR:",
            repr(e),
            flush=True
        )


# ============================================================
# PREDICTION
# ============================================================

@router.post("/")
async def predict(

    patient_id: str = Form(...),

    clinical_text: str = Form(...),

    xray: UploadFile = File(...),

    ecg: UploadFile = File(...),

    current_user=Depends(
        get_current_user
    )
):

    print()
    print("=" * 70)
    print("CARDIOPE-AI PREDICTION REQUEST")
    print("=" * 70)
    print(flush=True)

    xray_path = None
    ecg_path = None

    xray_filename = None
    ecg_filename = None

    # ========================================================
    # AUTH
    # ========================================================

    if not current_user:

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    if current_user.get("role") != "doctor":

        raise HTTPException(
            status_code=403,
            detail="Doctor authorization required."
        )

    doctor_id = current_user.get(
        "user_id"
    )

    if not doctor_id:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token."
        )

    print(
        ">>> DOCTOR:",
        doctor_id,
        flush=True
    )

    # ========================================================
    # PATIENT
    # ========================================================

    patient = patients_collection.find_one(
        {
            "patient_id": patient_id,
            "doctor_id": doctor_id
        }
    )

    if not patient:

        raise HTTPException(
            status_code=404,
            detail="Patient not found."
        )

    print(
        ">>> PATIENT:",
        patient.get("name"),
        flush=True
    )

    print(
        ">>> PATIENT ID:",
        patient_id,
        flush=True
    )

    # ========================================================
    # CLINICAL INFORMATION
    # ========================================================

    clinical_text = (
        clinical_text or ""
    ).strip()

    if not clinical_text:

        raise HTTPException(
            status_code=400,
            detail="Clinical information is required."
        )

    print(
        ">>> CLINICAL TEXT LENGTH:",
        len(clinical_text),
        flush=True
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    try:

        # ----------------------------------------------------
        # X-RAY
        # ----------------------------------------------------

        (
            xray_filename,
            xray_path
        ) = await save_uploaded_file(

            xray,

            XRAY_UPLOAD_DIR,

            {
                "image/jpeg",
                "image/jpg",
                "image/png"
            },

            {
                ".jpg",
                ".jpeg",
                ".png"
            },

            "xray"
        )

        # ----------------------------------------------------
        # ECG
        # ----------------------------------------------------

        (
            ecg_filename,
            ecg_path
        ) = await save_uploaded_file(

            ecg,

            ECG_UPLOAD_DIR,

            {
                "image/jpeg",
                "image/jpg",
                "image/png"
            },

            {
                ".jpg",
                ".jpeg",
                ".png"
            },

            "ecg"
        )

    except HTTPException:

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise

    except Exception as e:

        print(
            ">>> FILE UPLOAD ERROR:",
            repr(e),
            flush=True
        )

        traceback.print_exc()

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded files."
        )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    if not xray_path or not os.path.exists(xray_path):

        cleanup_file(ecg_path)

        raise HTTPException(
            status_code=500,
            detail="X-Ray file was not saved correctly."
        )

    if not ecg_path or not os.path.exists(ecg_path):

        cleanup_file(xray_path)

        raise HTTPException(
            status_code=500,
            detail="ECG file was not saved correctly."
        )

    print()
    print(">>> X-RAY PATH:")
    print(xray_path, flush=True)

    print()
    print(">>> ECG PATH:")
    print(ecg_path, flush=True)

    # ========================================================
    # MODEL
    # ========================================================

    try:

        print()
        print("=" * 70)
        print(">>> ABOUT TO RUN FUSION MODEL")
        print("=" * 70)
        print(flush=True)

        result = predict_fusion(

            clinical_text=clinical_text,

            image_path=xray_path,

            ecg_path=ecg_path
        )

        print()
        print("=" * 70)
        print(">>> FUSION MODEL FINISHED")
        print("=" * 70)
        print(flush=True)

        print(
            ">>> MODEL RESULT TYPE:",
            type(result),
            flush=True
        )

        print(
            ">>> MODEL RESULT:",
            result,
            flush=True
        )

    except Exception as e:

        print()
        print("=" * 70)
        print(">>> FUSION MODEL ERROR")
        print("=" * 70)

        print(
            "ERROR:",
            repr(e),
            flush=True
        )

        traceback.print_exc()

        print("=" * 70)
        print(flush=True)

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise HTTPException(

            status_code=500,

            detail=(
                "Fusion model failed: "
                f"{str(e)}"
            )
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise HTTPException(

            status_code=500,

            detail=(
                "Fusion model returned "
                "an invalid result."
            )
        )

    # ========================================================
    # GET RESULT
    # ========================================================

    abnormal_probability = result.get(
        "abnormal_probability"
    )

    risk_percentage = result.get(
        "risk_percentage"
    )

    assessment = result.get(
        "assessment"
    )

    # ========================================================
    # PROBABILITY
    # ========================================================

    try:

        if abnormal_probability is not None:

            abnormal_probability = float(
                abnormal_probability
            )

        elif risk_percentage is not None:

            abnormal_probability = (
                float(risk_percentage)
                / 100.0
            )

        else:

            raise ValueError(
                "Model did not return probability."
            )

    except Exception as e:

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        print(
            ">>> PROBABILITY ERROR:",
            repr(e),
            flush=True
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Invalid probability returned "
                "by fusion model."
            )
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    if (
        abnormal_probability > 1
        and abnormal_probability <= 100
    ):

        abnormal_probability /= 100.0

    if not (
        0.0
        <= abnormal_probability
        <= 1.0
    ):

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise HTTPException(

            status_code=500,

            detail=(
                "Risk probability must be "
                "between 0 and 1."
            )
        )

    risk_percentage = round(
        abnormal_probability * 100,
        2
    )

    # ========================================================
    # ASSESSMENT
    # ========================================================

    if (
        not isinstance(
            assessment,
            str
        )
        or not assessment.strip()
    ):

        assessment = (

            "HIGH RISK"

            if abnormal_probability >= 0.5

            else

            "LOW RISK"
        )

    assessment = assessment.strip()

    # ========================================================
    # ECG RESULT
    # ========================================================

    ecg_analysis = result.get(
        "ecg_analysis"
    )

    print()
    print(">>> ECG ANALYSIS:")
    print(
        ecg_analysis,
        flush=True
    )

    # ========================================================
    # NORMALIZED RESULT
    # ========================================================

    normalized_result = {

        "abnormal_probability":
            round(
                abnormal_probability,
                4
            ),

        "risk_percentage":
            risk_percentage,

        "assessment":
            assessment,

        "ecg_analysis":
            ecg_analysis
    }

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    prediction_id = str(
        uuid.uuid4()
    )

    prediction_data = {

        "prediction_id":
            prediction_id,

        "patient_id":
            patient_id,

        "doctor_id":
            doctor_id,

        "patient_name":
            patient.get("name"),

        "patient_age":
            patient.get("age"),

        "patient_gender":
            patient.get("gender"),

        "clinical_text":
            clinical_text,

        "xray_filename":
            xray_filename,

        "xray_path":
            xray_path,

        "ecg_filename":
            ecg_filename,

        "ecg_path":
            ecg_path,

        "ecg_uploaded":
            True,

        "result":
            normalized_result,

        "created_at":
            datetime.now(
                timezone.utc
            )
    }

    try:

        predictions_collection.insert_one(
            prediction_data
        )

        print(
            ">>> PREDICTION SAVED TO DATABASE",
            flush=True
        )

    except Exception as e:

        print(
            ">>> DATABASE SAVE ERROR:",
            repr(e),
            flush=True
        )

        traceback.print_exc()

        cleanup_file(xray_path)
        cleanup_file(ecg_path)

        raise HTTPException(

            status_code=500,

            detail=(
                "Prediction completed but "
                "could not be saved."
            )
        )

    # ========================================================
    # SUCCESS LOG
    # ========================================================

    print()
    print("=" * 70)
    print("PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Patient:",
        patient.get("name"),
        flush=True
    )

    print(
        "Risk:",
        f"{risk_percentage:.2f}%",
        flush=True
    )

    print(
        "Assessment:",
        assessment,
        flush=True
    )

    print(
        "X-Ray:",
        xray_filename,
        flush=True
    )

    print(
        "ECG:",
        ecg_filename,
        flush=True
    )

    print(
        "Prediction ID:",
        prediction_id,
        flush=True
    )

    print("=" * 70)
    print(flush=True)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "message":
            "CardioPE-AI prediction completed",

        "prediction_id":
            prediction_id,

        "patient_id":
            patient_id,

        "patient_name":
            patient.get("name"),

        "doctor_id":
            doctor_id,

        "role":
            current_user.get(
                "role"
            ),

        "ecg_uploaded":
            True,

        "ecg_filename":
            ecg_filename,

        "result":
            normalized_result
    }