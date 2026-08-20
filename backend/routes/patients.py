import uuid

from fastapi import APIRouter, Depends, HTTPException

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

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    # --------------------------------------------------------
    # DOCTOR AUTHORIZATION
    # --------------------------------------------------------

    if current_user.get("role") != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Doctor authorization required."
        )

    doctor_id = current_user.get("user_id")

    if not doctor_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid doctor account."
        )

    # --------------------------------------------------------
    # CREATE PATIENT ID
    # --------------------------------------------------------

    patient_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # PATIENT DOCUMENT
    # --------------------------------------------------------

    patient_data = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "name": patient.name.strip(),
        "age": patient.age,
        "gender": patient.gender,
        "blood_group": patient.blood_group,
        "phone": patient.phone
    }

    # --------------------------------------------------------
    # SAVE TO MONGODB
    # --------------------------------------------------------

    result = patients_collection.insert_one(patient_data)

    # IMPORTANT:
    # PyMongo adds "_id" to the dictionary automatically.
    # Remove it before returning the response to FastAPI.

    patient_data.pop("_id", None)

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "message": "Patient created successfully",
        "patient": patient_data,
        "patient_id": patient_id,
        "doctor_id": doctor_id
    }


# ============================================================
# GET ALL PATIENTS
# ============================================================

@router.get("/")
async def get_patients(
    current_user=Depends(get_current_user)
):

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    # --------------------------------------------------------
    # DOCTOR AUTHORIZATION
    # --------------------------------------------------------

    if current_user.get("role") != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Doctor authorization required."
        )

    doctor_id = current_user.get("user_id")

    # --------------------------------------------------------
    # GET PATIENTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    # --------------------------------------------------------
    # DOCTOR AUTHORIZATION
    # --------------------------------------------------------

    if current_user.get("role") != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Doctor authorization required."
        )

    doctor_id = current_user.get("user_id")

    # --------------------------------------------------------
    # FIND PATIENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return patient


# ============================================================
# GET PATIENT PREDICTION HISTORY
# ============================================================

@router.get("/{patient_id}/predictions")
async def get_patient_predictions(
    patient_id: str,
    current_user=Depends(get_current_user)
):

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    # --------------------------------------------------------
    # DOCTOR AUTHORIZATION
    # --------------------------------------------------------

    if current_user.get("role") != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Doctor authorization required."
        )

    doctor_id = current_user.get("user_id")

    # --------------------------------------------------------
    # VERIFY PATIENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GET PREDICTIONS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "patient_id": patient_id,
        "patient_name": patient.get("name"),
        "predictions": predictions,
        "count": len(predictions)
    }