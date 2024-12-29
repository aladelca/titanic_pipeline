### TODO: Pendiente revisar error API
import pickle
from cleaning_name import CleaningName
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


class ModelInput(BaseModel):
    Age: float
    Pclass: int
    Fare: float
    Sex: str
    Name: str


app = FastAPI(
    title="API de Predicciones",
    description="Esta API recibe datos de entrada y retorna la predicción de un modelo en pickle.",
    version="1.0.0"
)


with open("final_model.pickle", "rb") as f:
    modelo = pickle.load(f)



@app.post("/predict")
def predict(data: ModelInput):
    
    features = [[
        data.Age,
        data.Pclass,
        data.Fare,
        data.Sex,
        data.Name
    ]]

    prediction = modelo.predict(features)
    

    return {
        "prediction": int(prediction[0])
    }