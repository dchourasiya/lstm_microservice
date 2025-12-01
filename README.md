# LSTM Temperature Forecasting Microservice

## 🌡️ Overview
This project demonstrates how a trained **LSTM (Long Short-Term Memory)** neural network can be converted into a microservice capable of real-time temperature forecasting.  
The API accepts **30 historical temperature values** and returns the **next predicted temperature**.

The backend is built using **FastAPI**, the model is implemented in **PyTorch**, and the application is deployed on **Render**, making it publicly accessible through a single link.



## 🚀 Live API Endpoint
**Base URL:**  
👉 https://lstm-microservice.onrender.com  

### Available Routes
- **GET /** – Confirms the API is running  
- **POST /predict** – Accepts JSON data and returns the temperature prediction  
- **POST /predict_web** – Web form interface for entering values manually  



## 🧠 How It Works
1. User provides **30 comma-separated temperature values**.  
2. The values are scaled using a pre-trained **MinMaxScaler**.  
3. The LSTM model (`lstm_temp_model.pth`) predicts the next temperature.  
4. The scaler reverses the transformation to return the prediction in °C.


## 🧪 Example Input
```json
{
  "values": [12.5, 13.1, 14.0, 15.2, 16.1, 15.8, 14.9, 13.7, 12.9, 12.1,
             11.8, 12.0, 12.3, 13.0, 13.8, 14.5, 15.0, 15.3, 15.1, 14.7,
             14.0, 13.4, 12.9, 12.2, 11.9, 12.0, 12.2, 12.5, 12.8, 13.0]
}
