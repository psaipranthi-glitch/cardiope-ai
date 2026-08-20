from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import router as auth_router
from backend.routes.patients import router as patients_router
from backend.routes.prediction import router as prediction_router
from backend.routes.patient_predictions import (
    router as patient_predictions_router
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="CardioPE-AI API",
    description="Multimodal Cardiac Intelligence",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    auth_router
)

app.include_router(
    patients_router
)

app.include_router(
    prediction_router
)

app.include_router(
    patient_predictions_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "CardioPE-AI API",
        "status": "online"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "CardioPE-AI"
    }