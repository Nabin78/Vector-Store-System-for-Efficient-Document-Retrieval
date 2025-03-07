from pymongo import MongoClient
import os
import json

cur_dir = os.getcwd()

def load_config_json(path:str=None):

    if path is None:
        path = "config.json" 

    os.chdir(cur_dir)
    with open(path, "r") as f:
        data = json.load(f)

    return data

mongo_uri = load_config_json()["MONGO_URI"]
client = MongoClient(mongo_uri)
db = client["Faiss_Vector_DB"]

if "text_files" not in db.list_collection_names():
    db.create_collection("text_files")

if "image_files" not in db.list_collection_names():
    db.create_collection("image_files")

if "index" not in db.list_collection_names():
    db.create_collection("index")

TEXT = db["text_files"]
IMAGE = db["image_files"]
INDEX = db["index"]