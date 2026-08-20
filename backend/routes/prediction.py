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


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


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


os.makedirs(XRAY_UPLOAD_DIR, exist_ok=True)
os.makedirs(ECG_UPLOAD_DIR, exist_ok=True)


async def save_uploaded_file(
    upload: UploadFile,
    directory: str,
    allowed_types: set,
    allowed_extensions: set,
    prefix: str
):

    content_type = (
        upload.content_type or ""
    ).lower()

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {prefix} file type."
        )

    original_filename = (
        upload.filename or ""
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {prefix} file extension."
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

    contents = await upload.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded {prefix} file is empty."
        )

    with open(file_path, "wb") as file:
        file.write(contents)

    print(
        f"[UPLOAD] {prefix}: "
        f"{filename} "
        f"({len(contents) / 1024 / 1024:.2f} MB)"
    )

    return filename, file_path


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

    print("\n" + "=" * 70)
    print("PREDICTION REQUEST RECEIVED")
    print("=" * 70)

    xray_path = None
    ecg_path = None

    try:

        # ====================================================
        # AUTH
        # ====================================================

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

        doctor_id = current_user.get("user_id")

        if not doctor_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token."
            )

        print("[1/8] Authentication OK")


        # ====================================================
        # PATIENT
        # ====================================================

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
            "[2/8] Patient OK:",
            patient.get("name")
        )


        # ====================================================
        # CLINICAL
        # ====================================================

        clinical_text = (
            clinical_text or ""
        ).strip()

        if not clinical_text:
            raise HTTPException(
                status_code=400,
                detail="Clinical information is required."
            )

        print(
            "[3/8] Clinical text OK:",
            len(clinical_text),
            "characters"
        )


        # ====================================================
        # X-RAY
        # ====================================================

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

        print(
            "[4/8] X-Ray saved:",
            xray_path
        )


        # ====================================================
        # ECG
        # ====================================================

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

        print(
            "[5/8] ECG saved:",
            ecg_path
        )


        # ====================================================
        # MODEL START
        # ====================================================

        print()
        print("=" * 70)
        print("[6/8] STARTING FUSION MODEL")
        print("=" * 70)

        print("Clinical:", len(clinical_text))
        print("X-Ray:", xray_path)
        print("ECG:", ecg_path)

        print(
            "Calling predict_fusion()..."
        )


        # ====================================================
        # FUSION MODEL
        # ====================================================

        result = predict_fusion(

            clinical_text=clinical_text,

            image_path=xray_path,

            ecg_path=ecg_path
        )


        # ====================================================
        # MODEL FINISHED
        # ====================================================

        print()
        print("=" * 70)
        print("[7/8] FUSION MODEL COMPLETED")
        print("=" * 70)

        print(
            "Result type:",
            type(result)
        )

        print(
            "Result:",
            result
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if not isinstance(result, dict):

            raise ValueError(
                "Fusion model returned invalid result."
            )

        abnormal_probability = result.get(
            "abnormal_probability"
        )

        risk_percentage = result.get(
            "risk_percentage"
        )

        assessment = result.get(
            "assessment"
        )


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

            raise ValueError(
                "Risk probability must be between 0 and 1."
            )


        risk_percentage = round(
            abnormal_probability * 100,
            2
        )


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
                else "LOW RISK"
            )

        assessment = assessment.strip()


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
                result.get(
                    "ecg_analysis"
                )
        }


        # ====================================================
        # SAVE
        # ====================================================

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


        predictions_collection.insert_one(
            prediction_data
        )


        print()
        print("=" * 70)
        print("[8/8] PREDICTION SAVED SUCCESSFULLY")
        print("=" * 70)

        print(
            "Prediction ID:",
            prediction_id
        )

        print(
            "Risk:",
            f"{risk_percentage:.2f}%"
        )

        print(
            "Assessment:",
            assessment
        )

        print("=" * 70)


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


    except HTTPException:

        print(
            "[HTTP ERROR]"
        )

        raise


    except Exception as e:

        print()
        print("=" * 70)
        print("CARDIOPE-AI PREDICTION ERROR")
        print("=" * 70)

        print(
            "ERROR:",
            repr(e)
        )

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(

            status_code=500,

            detail=f"Prediction failed: {str(e)}"
        )


    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        for path in (
            xray_path,
            ecg_path
        ):

            try:

                if (
                    path
                    and os.path.exists(path)
                ):

                    os.remove(path)

                    print(
                        "[CLEANUP] Removed:",
                        path
                    )

            except Exception as cleanup_error:

                print(
                    "[CLEANUP ERROR]:",
                    repr(cleanup_error)
                )