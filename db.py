from unicodedata import name
import pymongo
from decouple import config

client = pymongo.MongoClient("mongodb+srv://keder:"+config('DB_PASS')+"@dataset.uul2s.mongodb.net/comment?retryWrites=true&w=majority")
db = client.comment 

def update_data(data):
    db.dataset.insert_one(data) 