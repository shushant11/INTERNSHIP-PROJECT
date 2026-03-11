
# AI-Based Solar Panel Predictive Maintenance System

## Overview

This project implements a **Machine Learning system that predicts whether a solar panel requires maintenance** using operational and environmental data.

Solar panels can experience efficiency loss due to factors such as dust accumulation, temperature variations, and electrical irregularities. This project uses machine learning to analyze these factors and predict maintenance needs.

The goal is to demonstrate how **AI can improve reliability and efficiency in renewable energy systems**.

---

## Problem Statement

Solar power systems require regular maintenance to maintain optimal energy output. Traditional maintenance approaches are reactive and often performed only after system performance declines.

This project proposes a **predictive maintenance model** that analyzes solar panel operational data to determine whether maintenance is required.

---

## Dataset

Dataset used:

solar_maintenance_data.csv

The dataset includes several parameters affecting solar panel performance.

### Features

Temperature
Irradiance
Voltage
Current
Humidity
Dust Level

### Target Variable

Maintenance Required

---

## Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Streamlit
Machine Learning

---

## Machine Learning Model

The model used in this project is:

Random Forest Classifier

Random Forest was selected because it:

• Handles nonlinear relationships well
• Works efficiently on tabular datasets
• Provides high predictive accuracy

---

## Project Workflow

Dataset
↓
Data Preprocessing
↓
Feature Selection
↓
Train/Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Deployment

---

## Web Application

A Streamlit web application is implemented to allow users to input solar panel parameters and predict whether maintenance is required.

Run the application using:

streamlit run app.py

---

## How to Run the Project

Clone the repository:

git clone https://github.com/shushant11/INTERNSHIP-PROJECT

Install dependencies:

pip install pandas numpy scikit-learn streamlit

Run the ML model:

python spower.py

Run the web application:

streamlit run app.py

---

## Applications

Solar Power Plants
Renewable Energy Monitoring
Predictive Maintenance Systems
Energy Efficiency Optimization

---

## Future Improvements

• Real-time IoT sensor integration
• Multiple model comparison
• Deployment to cloud platforms
• Dashboard visualization

---

## Author

Shushant Kumar
