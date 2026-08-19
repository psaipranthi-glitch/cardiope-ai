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


router = APIRouter(
    prefix="/patient-predictions",
    tags=["Patient Predictions"]
)


# ============================================================
# PATIENT HISTORY
# ============================================================

@router.get("/{patient_id}")
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
            "patient_id":
                patient_id,

            "doctor_id":
                doctor_id
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
                "patient_id":
                    patient_id,

                "doctor_id":
                    doctor_id
            },
            {
                "_id": 0
            }
        ).sort(
            "created_at",
            -1
        )
    )


    for prediction in predictions:

        if "created_at" in prediction:

            prediction["created_at"] = (
                prediction["created_at"]
                .isoformat()
            )


    return {

        "patient_id":
            patient_id,

        "patient_name":
            patient.get("name"),

        "predictions":
            predictions
    }


# ============================================================
# ALL PREDICTIONS
# ============================================================

@router.get("/")
async def get_all_predictions(
    current_user=Depends(get_current_user)
):

    if not current_user:

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )


    doctor_id = current_user.get(
        "user_id"
    )


    predictions = list(
        predictions_collection.find(
            {
                "doctor_id":
                    doctor_id
            },
            {
                "_id": 0
            }
        ).sort(
            "created_at",
            -1
        )
    )


    for prediction in predictions:

        if "created_at" in prediction:

            prediction["created_at"] = (
                prediction["created_at"]
                .isoformat()
            )


    return {
        "predictions": predictions
    }