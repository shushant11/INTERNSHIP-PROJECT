
# AI-Based Solar Panel Predictive Maintenance System

## Project Overview

This project implements a **Machine Learning model to predict maintenance requirements for solar panels** based on operational and environmental data.

Solar power systems can experience performance degradation due to factors such as dust accumulation, temperature fluctuations, and equipment wear. Predictive maintenance helps detect potential issues before failure occurs.

This project demonstrates how **machine learning can be applied to renewable energy systems to improve efficiency and reduce downtime.**

---

## Problem Statement

Solar panels require periodic maintenance to maintain optimal efficiency. Traditional maintenance methods are reactive and performed after performance drops.

The objective of this project is to develop a **predictive maintenance model** that can determine whether a solar panel requires maintenance based on sensor and environmental data.

---

## Dataset

Dataset used:
solar_maintenance_data.csv

The dataset contains operational parameters of solar panels.

### Features

Temperature
Irradiance
Voltage
Current
Humidity
Dust Level

### Target Variable

Maintenance Required (Yes / No)

---

## Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Matplotlib
Machine Learning

---

## Machine Learning Model

The model used in this project is:

Random Forest Classifier

Random Forest is chosen because it provides:

• High accuracy
• Robustness to noise
• Ability to handle nonlinear relationships

---

## Machine Learning Workflow

Dataset
↓
Data Preprocessing
↓
Feature Selection
↓
Train-Test Split
↓
Model Training
↓
Model Evaluation
↓
Model Saving

---

## Model Performance

Accuracy achieved: ~85% – 90% depending on dataset split.

Evaluation metrics used:

Accuracy Score
Confusion Matrix

---

## How to Run the Project

Clone the repository:

git clone https://github.com/shushant11/INTERNSHIP-PROJECT

Install dependencies:

pip install pandas numpy scikit-learn

Run the program:

python spower.py

---

## Applications

Solar Power Plants
Renewable Energy Monitoring Systems
Predictive Maintenance Systems
Energy Efficiency Optimization

---

## Future Improvements

• Add real-time IoT sensor data integration
• Build a web interface using Streamlit
• Compare multiple machine learning models
• Deploy the model as an API

---

## Author

Shushant Kumar
