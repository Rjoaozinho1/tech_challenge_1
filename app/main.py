from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

class ModelInput(BaseModel):
    age: int  # Idade do paciente (numérico)
    sex: int  # Sexo do paciente (0 = female, 1 = male)
    bmi: int  # Índice de Massa Corporal (IMC) categorizado
    children: int  # Número de filhos
    smoker: int  # Fumante (0 = não fumante, 1 = fumante)
    region: int  # Região (0 = northeast, 1 = northwest, 2 = southeast, 3 = southwest)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pipeline_rf.joblib")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        data = [[input_data.age, input_data.sex, input_data.bmi, input_data.children, input_data.smoker, input_data.region]]
        prediction = model.predict(data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))