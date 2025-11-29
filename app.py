import torch
import torch.nn as nn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
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
# FastAPI app
# ------------------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>LSTM Temperature Prediction</h2>
    <p>Enter 30 comma-separated scaled values:</p>

    <form action="/predict_web" method="post">
        <textarea name="sequence" rows="5" cols="60"></textarea><br><br>
        <input type="submit" value="Predict Temperature">
    </form>
    """

@app.post("/predict_web", response_class=HTMLResponse)
def predict_web(sequence: str = Form(...)):
    try:
        values = [float(x.strip()) for x in sequence.split(",")]

        if len(values) != 30:
            return "<h3>Error: You must enter exactly 30 values.</h3>"

        seq = np.array(values).reshape(1, 30, 1)
        seq_tensor = torch.tensor(seq, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(seq_tensor).numpy()

        pred_value = scaler.inverse_transform(pred_scaled)[0][0]

        return f"<h3>Predicted Temperature: {pred_value:.4f} °C</h3>"

    except:
        return "<h3>Error: Invalid input format. Please enter numbers separated by commas.</h3>"
