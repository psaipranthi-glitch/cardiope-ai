from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# PATIENT
# ============================================================

class PatientCreate(BaseModel):

    name: str = Field(
        ...,
        min_length=1
    )

    age: int = Field(
        ...,
        ge=0,
        le=120
    )

    gender: str

    blood_group: Optional[str] = None

    phone: Optional[str] = None


class PatientResponse(BaseModel):

    patient_id: str

    doctor_id: str

    name: str

    age: int

    gender: str

    blood_group: Optional[str] = None

    phone: Optional[str] = None