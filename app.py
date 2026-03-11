import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("solar_model.pkl","rb"))

st.title("Solar Panel Maintenance Predictor")

st.write("Enter solar panel parameters to predict maintenance requirement")

temperature = st.number_input("Temperature")
irradiance = st.number_input("Irradiance")
voltage = st.number_input("Voltage")
current = st.number_input("Current")
humidity = st.number_input("Humidity")
dust = st.number_input("Dust Level")

if st.button("Predict Maintenance"):

    input_data = np.array([[temperature, irradiance, voltage, current, humidity, dust]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Maintenance Required")
    else:
        st.success("No Maintenance Required")