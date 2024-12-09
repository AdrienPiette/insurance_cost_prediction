import streamlit as st
import pickle
import pandas as pd

# Load the trained model and encoder
try:
    model = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\model_training\best_model.pkl', 'rb'))
    encoder = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\model_training\label_encoders.pkl', 'rb'))
    scaler = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\model_training\scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model, encoder, or scaler: {e}")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Health Insurance Prediction", page_icon="🏠", layout="wide")

# Page title and description
st.title("Health Insurance Prediction")
st.write("""
    This is a simple web app to predict the health insurance charges based on the data provided by the user.
    The model was trained using the [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle.
""")

# Collect user input
st.write("Please enter the following information to get the prediction of health insurance charges:")
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

# Prepare input data as a DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}])

# Separate categorical columns
categorical_columns = ['sex', 'smoker', 'region']


