from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

app = FastAPI()

class ModelInput(BaseModel):
    age: int  = Field(description="Idade do paciente (numérico)")
    sex: int  = Field(description="Sexo do paciente (0 = female, 1 = male)")
    bmi: float  = Field(description="Índice de Massa Corporal (IMC) categorizado")
    children: int  = Field(description="Número de filhos")
    smoker: int = Field(description="Fumante (0 = não fumante, 1 = fumante)")
    region: int = Field(description="Região (0 = northeast, 1 = northwest, 2 = southeast, 3 = southwest)")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "pipeline_rf.joblib")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        data = [
            [
                input_data.age,
                input_data.sex,
                input_data.bmi,
                input_data.children,
                input_data.smoker,
                input_data.region
            ]
        ]

        prediction = model.predict(data)

        return {"charges": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))