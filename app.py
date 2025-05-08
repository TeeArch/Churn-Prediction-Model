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

# Initialize scaler
scaler = StandardScaler()

# File Upload Section
st.subheader("Upload your CSV file for prediction")
uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

def predict_churn(data):
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    return (predictions >= 0.5).astype(int)

# If CSV File is Uploaded
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview", data.head())

        # Selecting relevant features
        features = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                         'Balance', 'NumOfProducts', 'HasCrCard',
                         'IsActiveMember', 'EstimatedSalary']]
        
        # One-Hot Encoding for Geography
        features['Germany'] = (features['Geography'] == 'Germany').astype(int)
        features['Spain'] = (features['Geography'] == 'Spain').astype(int)
        features.drop(columns=['Geography'], inplace=True)
        
        # One-Hot Encoding for Gender
        features['Male'] = (features['Gender'] == 'Male').astype(int)
        features.drop(columns=['Gender'], inplace=True)

        # Scaling the data
        scaler.fit(features)
        predictions = predict_churn(features)
        data['Churn_Prediction'] = predictions

        st.write("### Prediction Results", data[['CreditScore', 'Age', 'Balance', 'Churn_Prediction']])
        st.success("Prediction completed!")

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Please upload a CSV file for batch prediction.")

# Divider for Manual Entry
st.markdown("---")
st.subheader("Or enter customer details manually")

with st.form("manual_input_form"):
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    Tenure = st.slider("Tenure", 0, 10, 5)
    Balance = st.number_input("Balance", value=50000.0)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", value=60000.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # Encoding categorical values (One-Hot)
    Germany = 1 if Geography == "Germany" else 0
    Spain = 1 if Geography == "Spain" else 0
    Male = 1 if Gender == "Male" else 0

    # Creating DataFrame for manual input
    manual_input = pd.DataFrame([[CreditScore, Germany, Spain, Male, Age, Tenure,
                                  Balance, NumOfProducts, HasCrCard, 
                                  IsActiveMember, EstimatedSalary]],
                                columns=['CreditScore', 'Germany', 'Spain', 'Male', 
                                         'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                                         'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    # Ensure scaler is fitted before scaling
    if not uploaded_file:
        scaler.fit([[650, 0, 0, 1, 35, 5, 50000, 1, 1, 0, 60000]])  # Default fit for manual input

    # Scale manual input using the same fitted scaler
    manual_input_scaled = scaler.transform(manual_input)
    prediction = model.predict(manual_input_scaled)
    
    st.success(f"Prediction: {'Churn' if prediction[0][0] >= 0.5 else 'No Churn'}")
