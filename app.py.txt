import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# ------------------------------
# Load scaler
# ------------------------------
scaler = joblib.load("scaler.save")

# ------------------------------
# Recreate LSTM model class
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out

# ------------------------------
# Load trained model
# ------------------------------
model = LSTMModel()
model.load_state_dict(torch.load("lstm_temp_model.pth", map_location=torch.device("cpu")))
model.eval()

# ------------------------------
# API Request Body
# ------------------------------
class SequenceInput(BaseModel):
    sequence: list   # must contain 30 scaled values

# ------------------------------
# FASTAPI App
# ------------------------------
app = FastAPI()

@app.get("/")
def home():
    return {"message": "LSTM Temperature Forecast API is running"}

@app.post("/predict")
def predict(data: SequenceInput):

    # Convert list → numpy → tensor
    seq = np.array(data.sequence).reshape(1, 30, 1)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)

    # Model prediction (scaled)
    with torch.no_grad():
        pred_scaled = model(seq_tensor).numpy()

    # Convert back to Celsius
    pred_value = scaler.inverse_transform(pred_scaled)[0][0]

    return {
        "prediction_celsius": float(pred_value)
    }
