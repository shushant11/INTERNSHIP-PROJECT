{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a6b3323b-d105-44b7-bc35-3b176fb0582a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "88f7d4a4-f7aa-46d7-a525-84fbeabcaca8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:22:34.867 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
      "2025-03-07 10:22:34.876 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
      "2025-03-07 10:22:34.878 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:22:36.698 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\singh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-03-07 10:22:36.699 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:22:36.702 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:22:36.748 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:22:36.750 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Load dataset\n",
    "@st.cache_data\n",
    "def load_data():\n",
    "    df = pd.read_csv(\"solar_maintenance_data.csv\")\n",
    "    return df\n",
    "\n",
    "df = load_data()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "67b7d68b-c053-4055-b990-674bd0ccfc1c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:23:20.907 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:20.909 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:20.910 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:21.428 Thread 'Thread-4': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:21.432 Thread 'Thread-4': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:23.266 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:23:23.267 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Train model if not already trained\n",
    "@st.cache_resource\n",
    "def train_model():\n",
    "    X = df[[\"Temperature_C\", \"Voltage_V\", \"Current_A\", \"Efficiency_%\", \"Dust_Level_%\"]]\n",
    "    y = df[\"Maintenance_Needed\"]\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "    \n",
    "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = model.predict(X_test)\n",
    "    accuracy = accuracy_score(y_test, y_pred)\n",
    "    return model, accuracy\n",
    "\n",
    "model, model_accuracy = train_model()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fe5ec20d-77ec-4b25-9186-b9b70b403e44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:24:49.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.796 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.797 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.798 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.799 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.800 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.801 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.802 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.803 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.804 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.805 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:24:49.807 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# Streamlit UI\n",
    "st.title(\"Predictive Maintenance for Solar Power Systems\")\n",
    "\n",
    "# Display dataset\n",
    "if st.checkbox(\"Show Dataset\"):\n",
    "    st.write(df.head())\n",
    "\n",
    "# Show dataset statistics\n",
    "if st.checkbox(\"Show Data Statistics\"):\n",
    "    st.write(df.describe())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "79906f88-0699-481a-abb0-c6b8c0ffe05d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:25:58.375 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:58.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:58.674 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.067 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.069 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.072 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.074 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.324 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.585 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.587 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.588 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.589 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:25:59.695 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:00.475 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:00.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Correlation heatmap\n",
    "st.subheader(\"Correlation Heatmap\")\n",
    "fig, ax = plt.subplots()\n",
    "sns.heatmap(df.corr(), annot=True, cmap=\"coolwarm\", ax=ax)\n",
    "st.pyplot(fig)\n",
    "\n",
    "# Histogram of efficiency\n",
    "st.subheader(\"Efficiency Distribution\")\n",
    "fig, ax = plt.subplots()\n",
    "sns.histplot(df[\"Efficiency_%\"], bins=30, kde=True, ax=ax)\n",
    "st.pyplot(fig)\n",
    "\n",
    "# Scatter plot of temperature vs efficiency\n",
    "st.subheader(\"Temperature vs Efficiency\")\n",
    "fig, ax = plt.subplots()\n",
    "sns.scatterplot(x=df[\"Temperature_C\"], y=df[\"Efficiency_%\"], hue=df[\"Maintenance_Needed\"], palette=\"coolwarm\", ax=ax)\n",
    "st.pyplot(fig)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3c03f3a0-6d5c-4128-a550-4ceb45b00908",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:26:44.014 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.017 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.019 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.020 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.022 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.023 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.024 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.025 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.026 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.027 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.029 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.030 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.032 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.034 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.035 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.036 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.036 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.037 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.039 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.040 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.041 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.042 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.043 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.044 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:26:44.044 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# User input for prediction\n",
    "st.sidebar.header(\"Input Parameters\")\n",
    "temp = st.sidebar.slider(\"Temperature (°C)\", 20, 60, 40)\n",
    "voltage = st.sidebar.slider(\"Voltage (V)\", 30, 50, 40)\n",
    "current = st.sidebar.slider(\"Current (A)\", 5, 15, 10)\n",
    "efficiency = st.sidebar.slider(\"Efficiency (%)\", 70, 95, 85)\n",
    "dust_level = st.sidebar.slider(\"Dust Level (%)\", 0, 100, 50)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "81d01506-93ed-4327-be15-846aa1a45244",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\singh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\utils\\validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n",
      "C:\\Users\\singh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\utils\\validation.py:2739: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
      "  warnings.warn(\n",
      "2025-03-07 10:27:26.821 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:26.823 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:26.826 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:26.827 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Predict maintenance need\n",
    "input_data = np.array([[temp, voltage, current, efficiency, dust_level]])\n",
    "prediction = model.predict(input_data)\n",
    "prediction_proba = model.predict_proba(input_data)\n",
    "\n",
    "st.subheader(\"Prediction Result\")\n",
    "if prediction[0] == 1:\n",
    "    st.error(f\"⚠️ Maintenance Required! (Confidence: {prediction_proba[0][1]:.2f})\")\n",
    "else:\n",
    "    st.success(f\"✅ No Maintenance Needed (Confidence: {prediction_proba[0][0]:.2f})\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "48aa78b5-94c7-4a45-8938-191c34fbee33",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-07 10:27:48.486 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.488 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.489 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.490 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.492 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.495 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.620 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.768 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-03-07 10:27:48.771 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display model accuracy\n",
    "st.subheader(\"Model Accuracy\")\n",
    "st.info(f\"🔍 The model accuracy is: {model_accuracy:.2f}\")\n",
    "\n",
    "# Feature importance visualization\n",
    "st.subheader(\"Feature Importance\")\n",
    "feature_importance = model.feature_importances_\n",
    "feature_names = [\"Temperature_C\", \"Voltage_V\", \"Current_A\", \"Efficiency_%\", \"Dust_Level_%\"]\n",
    "fig, ax = plt.subplots()\n",
    "sns.barplot(x=feature_importance, y=feature_names, ax=ax)\n",
    "st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81d17053-fe07-4a1a-a066-0fe904b6435c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
