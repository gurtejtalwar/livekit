from mongoengine import connect
import os

from app.shared.settings import get_settings

settings = get_settings()

MONGO_DB = settings.MONGO_DB

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/itsbot-db")

connect(
    db=settings.MONGO_DB,
    host=settings.MONGO_URI,
    alias="default",  # this is what MongoEngine is complaining about
)