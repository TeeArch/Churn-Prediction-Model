import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained ANN model
model = load_model('model.h5')

# Streamlit App Configuration
st.set_page_config(page_title="Churn Prediction with ANN", layout="centered")
st.title("📊 Churn Prediction with ANN")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

def predict_churn(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    predictions = model.predict(data_scaled)
    return (predictions >= 0.5).astype(int)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", data.head())

        features = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard',
                         'IsActiveMember', 'EstimatedSalary']]
        
        features['Geography'] = features['Geography'].astype('category').cat.codes
        features['Gender'] = features['Gender'].map({'Male': 1, 'Female': 0})

        predictions = predict_churn(features)
        data['Churn_Prediction'] = predictions

        st.write("### Prediction Results", data[['CreditScore', 'Age', 'Balance', 'Churn_Prediction']])
        st.success("Prediction completed!")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Please upload a CSV file.")
