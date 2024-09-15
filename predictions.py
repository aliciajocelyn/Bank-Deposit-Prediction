# Nama : Alicia Jocelyn Siahaya
# NIM : 2602072552

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model and all encoders from pickle
model = joblib.load('SVM_model.pkl')
job_replacer = joblib.load('job_replacement.pkl')
contact_encoder = joblib.load('contact_encoder.pkl')
categorical_encoder = joblib.load('categorical_encoder.pkl')
day_encoder = joblib.load('day_encoder.pkl')
month_encoder = joblib.load('month_encoder.pkl')
default_encoder = joblib.load('default_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input data schema for the API
class Deposit(BaseModel):
    age: float
    default: int
    month: int
    day_of_week: int
    duration: float
    campaign: float
    pdays: float
    previous: float
    job: str
    marital: str
    education: str
    housing: str
    loan: str
    contact: str
    poutcome: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

# Prediction endpoint
@app.post('/predict')
def predict(Bank: Deposit):
    data = Bank.dict()  # Convert input to a dictionary

    # Replace rare job values with 'others'
    job = data['job']
    if job in job_replacer['job']:
        data['job'] = 'others'
    
    # Prepare the categorical and numeric features
    categorical_features = np.array([[data['job'], data['marital'], data['education'], data['housing'], data['loan'], data['contact'], data['poutcome']]])
    day_feature = np.array([[data['day_of_week']]])
    month_feature = np.array([[data['month']]])
    default_feature = np.array([[data['default']]])
    numeric_features = np.array([[data['age'], data['duration'], data['campaign'], data['pdays'], data['previous']]])

    # Apply encoding and scaling
    encoded_categorical = categorical_encoder.transform(categorical_features)
    encoded_day = day_encoder.transform(day_feature)
    encoded_month = month_encoder.transform(month_feature)
    encoded_default = default_encoder.transform(default_feature)
    data['contact'] = data['contact'].replace(contact_encoder['contact'])
    scaled_numeric = scaler.transform(numeric_features)
    
    # Concatenate all features
    features = np.hstack([scaled_numeric, encoded_default, encoded_month, encoded_day, encoded_categorical])
    
    # Predict using the SVM model
    prediction = model.predict(features)
    
    return {'prediction': prediction[0]}  # Return the prediction result