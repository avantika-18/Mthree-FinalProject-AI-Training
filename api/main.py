import mlflow
from fastapi import FastAPI, HTTPException
from api.schemas import HouseFeatures
import pandas as pd
import logging
import sqlite3
import time
import os
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import threading

app = FastAPI()

# Ensure logs and db directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('db', exist_ok=True)

# Logging setup
logging.basicConfig(filename='logs/app.log', level=logging.INFO)

# Metrics
REQUESTS = Counter('requests_total', 'Total requests')
LATENCY = Histogram('request_latency_seconds', 'Request latency')

# Set tracking URI to match the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model using the alias @production
MODEL_URI = "models:/California_Housing_Prediction@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Thread-local storage for SQLite connection
local = threading.local()

def get_db_connection():
    if not hasattr(local, 'conn'):
        local.conn = sqlite3.connect('db/logs.db', check_same_thread=False)
        local.conn.execute('''CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, latency REAL, features TEXT, prediction REAL, error TEXT)''')
    return local.conn

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
@LATENCY.time()
def predict(features: HouseFeatures):
    REQUESTS.inc()
    start_time = time.time()
    conn = get_db_connection()  # Create or get thread-local connection
    try:
        # Convert Pydantic model to dictionary
        feature_dict = features.model_dump()
        
        # Create DataFrame with only the features the model was trained on
        model_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        data = pd.DataFrame([{k: feature_dict[k] for k in model_features if k in feature_dict}])
        
        # Ensure all columns are numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
        
        # Predict using the model
        prediction = model.predict(data)[0]
        latency = time.time() - start_time
        
        # Log to file and SQLite
        logging.info(f"Features: {features}, Prediction: {prediction}, Latency: {latency}")
        conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", 
                     (time.strftime("%Y-%m-%d %H:%M:%S"), latency, str(features), prediction, None))
        conn.commit()
        
        return {"prediction": float(prediction)}
    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        logging.error(f"Error: {error_msg}")
        conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", 
                     (time.strftime("%Y-%m-%d %H:%M:%S"), latency, str(features), None, error_msg))
        conn.commit()
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if 'conn' in locals():
            conn.close()  # Close only if connection was created

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())