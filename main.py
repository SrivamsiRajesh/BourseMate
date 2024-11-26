from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from polygon import RESTClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

# Initialize FastAPI app
app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Polygon API Key (replace with your actual key)
API_KEY = "Zp7hFhaLuHUYViMTJnyxakcEoKAmVfiY"
client = RESTClient(API_KEY)

# Load model and scaler (these will be trained separately)
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.npy"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = np.load(SCALER_PATH, allow_pickle=True).item()
else:
    model = None
    scaler = None

# Fetch stock data from Polygon
def fetch_stock_data(symbol, start_date, end_date):
    try:
        aggs = client.get_aggs(symbol, 1, "day", start_date, end_date)
        df = pd.DataFrame([agg.__dict__ for agg in aggs])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['close']]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

# Predict next day price
def predict_price(symbol):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or scaler not found. Train and upload them first.")

    # Fetch recent data
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    recent_data = fetch_stock_data(symbol, "2023-01-01", today)

    if recent_data is None or recent_data.empty:
        raise HTTPException(status_code=404, detail="No data found for the given symbol.")

    # Prepare data for prediction
    look_back = 60
    recent_closing_prices = recent_data['close'].values[-look_back:]
    scaled_recent_data = scaler.transform(recent_closing_prices.reshape(-1, 1))

    # Predict
    X_input = np.array([scaled_recent_data])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    predicted_scaled_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled_price)

    return predicted_price[0][0]  # Return predicted mean price

# Route: Home Page (HTML)
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/index.html", "r") as f:
        return f.read()

# Route: Predict Endpoint
@app.post("/predict")
def predict(symbol: str):
    try:
        predicted_price = predict_price(symbol.upper())
        return {"symbol": symbol, "predicted_price": round(predicted_price, 2)}
    except HTTPException as e:
        return {"error": e.detail}
