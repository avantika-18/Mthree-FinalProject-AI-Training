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

app = FastAPI()

# Ensure logs and db directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('db', exist_ok=True)

# Logging setup
logging.basicConfig(filename='logs/app.log', level=logging.INFO)
conn = sqlite3.connect('db/logs.db')
conn.execute('''CREATE TABLE IF NOT EXISTS logs (timestamp TEXT, latency REAL, features TEXT, prediction REAL, error TEXT)''')

# Metrics (optional Prometheus)
REQUESTS = Counter('requests_total', 'Total requests')
LATENCY = Histogram('request_latency_seconds', 'Request latency')

# Set tracking URI to match the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure this matches your MLflow server address

# Load model (assume registered as Production)
MODEL_URI = "models:/California_Housing_Prediction@production"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
@LATENCY.time()
def predict(features: HouseFeatures):
    REQUESTS.inc()
    start_time = time.time()
    try:
        data = pd.DataFrame([features.model_dump()])
        prediction = model.predict(data)[0]
        latency = time.time() - start_time
        
        # Log to file and SQLite
        logging.info(f"Features: {features}, Prediction: {prediction}, Latency: {latency}")
        conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", 
                     (time.strftime("%Y-%m-%d %H:%M:%S"), latency, str(features), prediction, None))
        conn.commit()
        
        return {"prediction": prediction}
    except Exception as e:
        latency = time.time() - start_time
        logging.error(f"Error: {str(e)}")
        conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", 
                     (time.strftime("%Y-%m-%d %H:%M:%S"), latency, str(features), None, str(e)))
        conn.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest())