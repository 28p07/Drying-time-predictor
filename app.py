import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# ------------------------------
# Define the model architecture
# ------------------------------
class RegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("Drying Time Forecasting")

# User Inputs
solar_radiation = st.number_input("Solar Radiation (W/m²)")
initial_moisture = st.number_input("Initial Moisture Content (%)")
final_moisture = st.number_input("Final Moisture Content (%)")
drying_rate = st.number_input("Drying Rate (kg/hr)")
chamber_temp = st.number_input("Drying Chamber Temperature (°C)")
velocity1 = st.number_input("Velocity Inside Chamber (m/s)")
humidity = st.number_input("Humidity Inside Chamber (%)")

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Drying Time"):

    # Load scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Prepare input
    new_point = np.array([[solar_radiation, initial_moisture, final_moisture,
                           drying_rate, chamber_temp, velocity1, humidity]])
    new_point_scaled = scaler.transform(new_point)
    input_tensor = torch.tensor(new_point_scaled, dtype=torch.float32)

    # Load model weights into defined architecture
    model = RegressionModel(num_features=7)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        drying_time = int(prediction.item())  # Convert to plain int

    st.success(f"Predicted Drying Time: {drying_time} minutes")
