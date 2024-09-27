from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

app = FastAPI()

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: int
    SibSp: int 
    Parch: int
    Embarked:str
   

@app.get("/")
def home():
   return("hello")

@app.get("/by_name/{name}")
def byname(name):
    return (f"Welcome {name}!")

@app.post("/predict")
def predict(data: Passenger):
    try:
        input_data = pd.DataFrame([{
            'Pclass': data.Pclass,
            'Sex':data.Sex,
            'Age':data.Age,
            'SibSp':data.SibSp,
            'Parch':data.Parch,
            'Embarked':data.Embarked,
        }])

        processed_data = preprocessor.transform(input_data)

        model_prediction = model.predict(processed_data)

        prediction_as_list = model_prediction.tolist()

        if model_prediction[0] > 0.5:
                result = "Survived"
        else:
            result = "Not Survived"

        return {"model's Prediction": result}


    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
