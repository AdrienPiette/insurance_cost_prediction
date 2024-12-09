import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load the model

model = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\model_training\best_model.pkl', 'rb'))
encoder = pickle.load(open(r'C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\model_training\encoder.pkl', 'rb'))


st.set_page_config(page_title="Health Insurance Prediction", page_icon="🏠", layout="wide"
)

st.title("Health Insurance Prediction")
st.write("""
    This is a simple web app to predict the health insurance charges based on the data provided by the user.
    The data is used to train a machine learning model that predicts the charges based on the input data.
    The model is trained using the [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle.
""")

#collecting user input

st.write("Please enter the following information to get the prediction of the health insurance charges.")
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input("BMI", min_value=0, max_value=100, value=20)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])





