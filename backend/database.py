import os

from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "cardiope_ai"
)


if not MONGO_URI:
    raise ValueError(
        "MONGO_URI is missing from .env"
    )


client = MongoClient(MONGO_URI)

db = client[DATABASE_NAME]


users_collection = db["users"]

patients_collection = db["patients"]

records_collection = db["medical_records"]

predictions_collection = db["predictions"]


def test_connection():

    client.admin.command("ping")

    return True