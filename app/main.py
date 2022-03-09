from typing import Optional
from fastapi import FastAPI, File, UploadFile
import requests
import shutil
import subprocess
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/count-berries")
def count_berries(image: UploadFile = File(...)):

	path = f"./app/etc/{image.filename}"
	with open(path, 'wb') as fileimg:
		fileimg.write(image.file.read())

	return subprocess.run(['python3.9',"./demo/demo.py", 
                           "--input", path, 
                           "--output", "grape_from_real", 
                           "--opts", "MODEL.DEVICE", "cpu",
                            "MODEL.WEIGHTS", "./training_dir/BoxInst_MS_R_50_1x/model_final.pth"],	
                          capture_output=True)
