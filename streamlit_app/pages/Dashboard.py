import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv(r"C:\Users\pieta\OneDrive\Bureau\insurance_prediction\insurance_cost_prediction\data\insurance.csv")

# Streamlit title and description
st.title("Dashboard")
st.subheader("Data Visualization")
st.write("This is a dashboard to visualize the data used to train the machine learning model. The data is used to predict the health insurance charges based on the input data provided by the user. The model is trained using the [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) from Kaggle.")

# Display the first few rows of the dataset
st.write(data.head())

# Show columns
st.write("The data contains the following columns:")
st.write(data.columns)

# Define custom colors (orange and teal)
color_map_sex = {'male': 'teal', 'female': 'orange'}
color_map_smoker = {'yes': 'teal', 'no': 'orange'}
color_map_region = {'southwest': 'teal', 'southeast': 'orange', 'northwest': 'teal', 'northeast': 'orange'}
color_map_children = {0: 'teal', 1: 'orange', 2: 'teal', 3: 'orange', 4: 'teal', 5: 'orange'}
color_map_age = {18: 'teal', 19: 'orange', 20: 'teal', 21: 'orange', 22: 'teal', 23: 'orange', 24: 'teal', 25: 'orange'}
color_map_bmi = {0: 'teal', 1: 'orange'}

# Plotting histograms for various features
fig = px.histogram(data, x='charges', color='sex', barmode='overlay', 
                   title='Charges by sex', color_discrete_map=color_map_sex)
st.plotly_chart(fig)

fig = px.histogram(data, x='charges', color='smoker', barmode='overlay', 
                   title='Charges by smoker', color_discrete_map=color_map_smoker)
st.plotly_chart(fig)

fig = px.histogram(data, x='charges', color='region', barmode='overlay', 
                   title='Charges by region', color_discrete_map=color_map_region)
st.plotly_chart(fig)

fig = px.histogram(data, x='charges', color='children', barmode='overlay', 
                   title='Charges by number of children', color_discrete_map=color_map_children)
st.plotly_chart(fig)

fig = px.histogram(data, x='charges', color='age', barmode='overlay', 
                   title='Charges by age', color_discrete_map=color_map_age)
st.plotly_chart(fig)


# Define custom colors (orange and teal)
color_map_bmi = {0: 'teal', 1: 'orange'}

# Group the data by BMI and calculate the mean of charges
data_agr = data.groupby('bmi')['charges'].mean().reset_index()

# Convert BMI to categories if needed for better visualization
data_agr['bmi_category'] = pd.cut(data_agr['bmi'], bins=10, labels=[f'{i}-{i+1}' for i in range(0, 40, 4)])

# Create the bar chart for average charges by BMI category
fig = px.bar(data_agr, x='bmi_category', y='charges', color='bmi_category', 
             title='Charges by BMI Category', color_discrete_map=color_map_bmi)
st.plotly_chart(fig)

# Label Encoding for categorical columns
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Correlation matrix
corr = round(data.corr(), 2)
fig_9 = px.imshow(
    corr, 
    color_continuous_scale='Viridis',  
    text_auto=True,  
    title='Correlation Matrix for Insurance Data'
)
fig_9.update_layout(
    template='plotly_white',
    title_font_size=16,
    title_x=0.5  
)
st.subheader("Correlation Matrix for Insurance Data")
st.plotly_chart(fig_9, use_container_width=True)

# Most Correlated Features with Charges
corr_charges = corr['charges'].sort_values(ascending=False)
corr_charges = corr_charges[1:]  # Remove the target variable

fig_10 = px.bar(
    corr_charges, 
    x=corr_charges.index, 
    y=corr_charges.values, 
    color=corr_charges.values,  
    color_continuous_scale='Viridis',  
    title='Most Correlated Features with Charges'
)
fig_10.update_layout(
    xaxis_title='Feature',
    yaxis_title='Correlation',
    coloraxis_colorbar=dict(title='Correlation'),
    template='plotly_white',
    title_font_size=16,
    title_x=0.5  
)
st.subheader("Most Correlated Features with Charges")
st.plotly_chart(fig_10, use_container_width=True)
