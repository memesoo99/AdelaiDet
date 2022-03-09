from typing import Optional, List
from fastapi import FastAPI, File, UploadFile,
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import csv
import requests
import shutil
import subprocess
import os

class Grape(BaseModel):
    filename: str
    hashkey: str
    berry: int

class GrapeList(BaseModel):
    data: List[Grape]

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/count-berries", response_model=UserIn)
def count_berries(quality_id: int, images: List[UploadFile]):
        path = []
        for image in images:
	    path.append(image.filename)
	    with open("./app/etc/"+path[-1], 'wb') as fileimg:
		fileimg.write(image.file.read())

	subprocess.run(['python3.9',"./demo/demo.py", 
                           "--input", " ".join(list(map(lambda x:"./app/etc/"+x, path))), 
                           "--code", str(quality_id),
                           "--output", "grape_from_real", 
                           "--opts", "MODEL.DEVICE", "cpu",
                            "MODEL.WEIGHTS", "./training_dir/BoxInst_MS_R_50_1x/model_final.pth"],	
                          capture_output=True)
        #df_result = pd.read_csv('../viz/results/'+str(quality_id)+'.csv') 
        
        result_csv = csv.reader(open('../viz/results/'+str(quality_id)+'.csv'))
    
        
        grape_list = []
        for p in path:
            with open("../viz/"+p, 'rb') as fileimg:
                response = requests.post('https://ipfs.infura.io:5001/api/v0/add',
                        files= {'file':fileimg} )
               # hashkey.append( response.json()['Hash'] )
                obj = Grape(filename=p, hashkey=response.json()['Hash'], berry=next(result)[-1])
                grape_list.append(obj)
        return GrapeList(data = grape_list)
