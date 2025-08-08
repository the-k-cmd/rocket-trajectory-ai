# %%
import streamlit as st
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
model = tf.keras.models.load_model('flightPathModel.h5', compile=False)

with open('columns.json') as f:
    columns = json.load(f)

st.title('🚀 Launch Path Prediction')

# %%
noseType = st.selectbox('Nose Type', ['Conical', 'Ogive', 'Triangular', 'Trapezoidal', 'Grid', 'None'])
finShape = st.selectbox('Fin Shape', ['Grid', 'Trapezoidal', 'Triangular', 'None'])
noseLength = st.number_input('Nose Length (m)', 0.0)
bodyLength = st.number_input('Body Length (m)', 0.0)
bodyDia = st.number_input('Body Diameter (m)', 0.0)
finCount = st.number_input('Fin Count', 0)
finArea = st.number_input('Fin Area (m²)', 0.0)
stageCount = st.number_input('Number of Stages', 1)
totalMass = st.number_input('Total Mass (kg)', 0.0)
fuelMass = st.number_input('Fuel Mass (kg)', 0.0)
thrust = st.number_input('Thrust (N)', 0.0)
burnTime = st.number_input('Burn Time (s)', 0.0)
isp = st.number_input('Specific Impulse (s)', 0.0)
drag = st.number_input('Drag Coefficient', 0.0)
angle = st.number_input('Launch Angle (°)', 0.0)
wind = st.number_input('Wind Speed (m/s)', 0.0)

# %%
def predict():
    inputData = pd.DataFrame([{
        'NoseType': noseType,
        'FinShape': finShape,
        'NoseLength': noseLength,
        'BodyLength': bodyLength,
        'BodyDiameter': bodyDia,
        'FinCount': finCount,
        'FinArea': finArea,
        'StageCount': stageCount,
        'TotalMass': totalMass,
        'FuelMass': fuelMass,
        'Thrust': thrust,
        'BurnTime': burnTime,
        'ISP': isp,
        'DragCoefficient': drag,
        'LaunchAngle': angle,
        'WindSpeed': wind,

        **{f'NoseType_{nt}': 1 if nt == noseType else 0
            for nt in ['Conical', 'Ogive', 'Triangular', 'Trapezoidal', 'Grid', 'None']},
        **{f'FinShape_{fs}': 1 if fs == finShape else 0
            for fs in ['Grid', 'Trapezoidal', 'Triangular', 'None']}
    }])

    inputData = inputData.reindex(columns=columns, fill_value=0)

    prediction = model.predict(inputData)[0]
    peakAlt, tToPeak, totalTime = prediction

    t = np.linspace(0, totalTime, 200)
    h = -4 * peakAlt / (tToPeak ** 2) * (t - tToPeak) ** 2 + peakAlt
    h = np.maximum(h, 0) 

    maxHrizontalDistance = 0.5 * thrust / totalMass * totalTime ** 2 * 0.001
    x = np.linspace(0, maxHrizontalDistance, 200)

    fig, ax = plt.subplots()
    ax.plot(x, h)
    ax.set_xlabel('Horizontal Distance (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Predicted Trajectory')
    st.pyplot(fig)

# %%
if st.button('Predict Flight Profile'):
    predict()


