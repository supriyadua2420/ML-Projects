from fastapi import FastAPI, File, UploadFile
import numpy as np
import joblib
from PIL import Image

app = FastAPI()
model = joblib.load("digital_model.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L").resize((28, 28))
    data = np.array(image).reshape(1, -1)/255.0
    pred = model.predict(data)
    return {"predicted_digit": int(pred[0])}
