import os
import sys
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="User Purchase Prediction API")

class PredictionInput(BaseModel):
    Gender: str
    Age: str
    Occupation: int
    City_Category: str
    Stay_In_Current_City_Years: str
    Marital_Status: int
    Product_Category_1: int
    Product_Category_2: float
    Product_Category_3: float
    Product_ID: str

# Paths for models and encoders (Update as necessary)
MODEL_PATH = "artifacts/models/model.pkl"
ENCODER_GENDER = "artifacts/encoders/le_gender.pkl"
ENCODER_CITY = "artifacts/encoders/le_city.pkl"
ENCODER_PRODUCT_FREQ = "artifacts/encoders/product_freq.pkl"

def load_object(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

model = load_object(MODEL_PATH)
le_gender = load_object(ENCODER_GENDER)
le_city = load_object(ENCODER_CITY)
product_freq = load_object(ENCODER_PRODUCT_FREQ)

@app.get("/")
def home():
    return {"message": "Welcome to User Purchase Prediction API"}

@app.post("/predict")
def predict(data: PredictionInput):
    if not model:
        # Return a dummy prediction if the model is not found
        return {"prediction": 9999.99, "warning": "Model not loaded. Returned dummy prediction."}
    
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Label Encoding
        if le_gender and 'Gender' in df.columns:
            # Handle unseen labels by defaulting or catching error
            try:
                df['Gender'] = le_gender.transform(df['Gender'])
            except:
                df['Gender'] = 0
                
        if le_city and 'City_Category' in df.columns:
            try:
                df['City_Category'] = le_city.transform(df['City_Category'])
            except:
                df['City_Category'] = 0
                
        # Age Mapping
        age_map = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}
        df["Age"] = df["Age"].map(age_map).fillna(0)
        
        # Stay_In_Current_City_Years mapping
        df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].replace("4+", 4).astype(int)
        
        # Frequency encoding for Product_ID
        if product_freq and 'Product_ID' in df.columns:
            df["Product_ID_FE"] = df["Product_ID"].map(product_freq).fillna(0)
        else:
            df["Product_ID_FE"] = 0
        
        df = df.drop(columns=['Product_ID'], errors='ignore')
        
        # Predict using the loaded model
        prediction = model.predict(df)
        
        return {"prediction": float(prediction[0])}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
