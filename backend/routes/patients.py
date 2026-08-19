import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from backend.auth import get_current_user
from backend.database import (
    patients_collection,
    predictions_collection
)
from backend.models import PatientCreate


router = APIRouter(
    prefix="/patients",
    tags=["Patients"]
)


# ============================================================
# CREATE PATIENT
# ============================================================

@router.post("/")
async def create_patient(
    patient: PatientCreate,
    current_user=Depends(get_current_user)
):

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

    patient_id = str(uuid.uuid4())

    patient_data = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "name": patient.name.strip(),
        "age": patient.age,
        "gender": patient.gender,
        "blood_group": patient.blood_group,
        "phone": patient.phone
    }

    patients_collection.insert_one(patient_data)

    return {
        "message": "Patient created successfully",
        "patient": patient_data,
        "patient_id": patient_id,
        "doctor_id": doctor_id
    }


# ============================================================
# GET PATIENTS
# ============================================================

@router.get("/")
async def get_patients(
    current_user=Depends(get_current_user)
):

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

    patients = list(
        patients_collection.find(
            {
                "doctor_id": doctor_id
            },
            {
                "_id": 0
            }
        )
    )

    return {
        "patients": patients
    }


# ============================================================
# GET SINGLE PATIENT
# ============================================================

@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user=Depends(get_current_user)
):

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

    patient = patients_collection.find_one(
        {
            "patient_id": patient_id,
            "doctor_id": doctor_id
        },
        {
            "_id": 0
        }
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found."
        )

    return patient


# ============================================================
# GET PATIENT PREDICTION HISTORY
# ============================================================

@router.get("/{patient_id}/predictions")
async def get_patient_predictions(
    patient_id: str,
    current_user=Depends(get_current_user)
):

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

    doctor_id = current_user.get("user_id")


    # ========================================================
    # VERIFY PATIENT BELONGS TO DOCTOR
    # ========================================================

    patient = patients_collection.find_one(
        {
            "patient_id": patient_id,
            "doctor_id": doctor_id
        },
        {
            "_id": 0
        }
    )

    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient not found."
        )


    # ========================================================
    # GET PREDICTIONS
    # ========================================================

    predictions = list(
        predictions_collection.find(
            {
                "patient_id": patient_id,
                "doctor_id": doctor_id
            },
            {
                "_id": 0
            }
        ).sort(
            "created_at",
            -1
        )
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "patient_id": patient_id,
        "patient_name": patient.get("name"),
        "predictions": predictions,
        "count": len(predictions)
    }
# ============================================================
# GET PATIENT PREDICTION HISTORY
# ============================================================

@router.get("/{patient_id}/predictions")
async def get_patient_predictions(
    patient_id: str,
    current_user=Depends(get_current_user)
):

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

    predictions = list(
        predictions_collection.find(
            {
                "patient_id": patient_id,
                "doctor_id": doctor_id
            },
            {
                "_id": 0
            }
        ).sort(
            "created_at",
            -1
        )
    )

    return {
        "predictions": predictions
    }