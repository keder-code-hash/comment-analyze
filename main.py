import imp
from prediction import *
import os

#### FastAPI import Part #####
from fastapi import FastAPI
app=FastAPI()
from typing import Optional
from pydantic import BaseModel
##### #####


#### api service ####
class Input(BaseModel):
    input_comment:str

@app.post("/analyze/")
def analyze_comment(comment:Input):
    result=predict_sense(comment.input_comment,api_mode=True)
    return result
#### #### 
# os.system('uvicorn main:app --reload ')
 
