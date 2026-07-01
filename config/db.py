# config/db.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("MONGO_URI is not set")

client = MongoClient(MONGO_URI)
db = client.get_database("Cluster0")

print("✅ MongoDB connected to Cluster0")
