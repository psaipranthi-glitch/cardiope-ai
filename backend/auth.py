from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr


# ============================================================
# CONFIG
# ============================================================

SECRET_KEY = "cardiope-ai-super-secret-key-change-later"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# ============================================================
# SECURITY
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

security = HTTPBearer()


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# ============================================================
# USER DATABASE
# ============================================================

users_db = {}


# ============================================================
# SCHEMAS
# ============================================================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: str = "doctor"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# ============================================================
# PASSWORD FUNCTIONS
# ============================================================

def hash_password(password: str) -> str:

    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:

    return pwd_context.verify(
        plain_password,
        hashed_password
    )


# ============================================================
# CREATE JWT
# ============================================================

def create_access_token(
    user_id: str,
    role: str
):

    expire = datetime.now(
        timezone.utc
    ) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


# ============================================================
# GET CURRENT USER
# ============================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        security
    )
):

    token = credentials.credentials

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id: Optional[str] = payload.get("sub")
        role: Optional[str] = payload.get("role")

        if not user_id:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )

        return {
            "user_id": user_id,
            "role": role
        }

    except JWTError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


# ============================================================
# REGISTER
# ============================================================

@router.post("/register")
def register(user: UserRegister):

    if user.email in users_db:

        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    user_id = str(
        uuid.uuid4()
    )

    users_db[user.email] = {

        "user_id": user_id,

        "email": user.email,

        "password": hash_password(
            user.password
        ),

        "role": user.role
    }

    return {

        "message": "Registration successful",

        "user_id": user_id,

        "role": user.role
    }


# ============================================================
# LOGIN
# ============================================================

@router.post("/login")
def login(user: UserLogin):

    # --------------------------------------------------------
    # EXISTING DEVELOPMENT DOCTOR ACCOUNT
    # --------------------------------------------------------

    if (
        user.email == "doctor@test.com"
        and user.password == "DoctorTest123"
    ):

        user_id = (
            "4f5c1c15-99df-4809-bd97-e21587e42be5"
        )

        role = "doctor"

    else:

        stored_user = users_db.get(
            user.email
        )

        if not stored_user:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        if not verify_password(
            user.password,
            stored_user["password"]
        ):

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        user_id = stored_user["user_id"]

        role = stored_user["role"]

    # --------------------------------------------------------
    # CREATE TOKEN
    # --------------------------------------------------------

    token = create_access_token(
        user_id,
        role
    )

    return {

        "message": "Login successful",

        "token": token,

        "user_id": user_id,

        "role": role
    }