# Churn Prediction with Artificial Neural Network (ANN)

This project is a Churn Prediction web application built using Streamlit and an Artificial Neural Network (ANN) model trained with TensorFlow/Keras. It allows users to either upload a CSV file of customer data for batch prediction or manually enter customer details for single prediction.

## Features
- Dual Input Mode:
  - Upload CSV file for batch predictions.
  - Manually enter customer details for single prediction.
- Real-time predictions using a pre-trained ANN model.
- One-Hot Encoding for categorical variables (Geography and Gender) matches model training.
- Intuitive user interface with clear instructions.

## Files in this Repository
- `app.py`: The main Streamlit app script.
- `model.h5`: The pre-trained Artificial Neural Network model.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Project documentation (this file).

## How to Run Locally
1. Clone the Repository:
   ```bash
   git clone https://github.com/TeeArch/Churn-Prediction-Model.git
   cd Churn-Prediction-Model
