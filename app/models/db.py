from mongoengine import connect
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/itsbot-db")

connect(
    db="itsbot-db",
    host=MONGO_URI,
    alias="default",  # this is what MongoEngine is complaining about
)